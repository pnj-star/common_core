import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import jwt
import pytest
from jwt.algorithms import RSAAlgorithm

from common_core.auth import AuthError, IdentityClaims, TokenVerifier, parse_bearer_token
from common_core.config import AuthConfig


def _token(secret: str, **claims: object) -> str:
    payload = {"exp": int(time.time()) + 3600, **claims}
    return jwt.encode(payload, secret, algorithm="HS256")


class _JwksHandler(BaseHTTPRequestHandler):
    """返回内存中动态更新的 JWKS JSON，供轮换测试使用。"""

    def do_GET(self) -> None:
        with self.server.jwks_lock:
            body = json.dumps({"keys": list(self.server.jwks)}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        return


def _rsa_signing_key(kid: str):
    """生成一把 RSA 签名私钥与对应的 JWK 条目。"""
    from cryptography.hazmat.primitives.asymmetric import rsa

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(RSAAlgorithm.to_jwk(private_key.public_key()))
    jwk.update({"kid": kid, "use": "sig", "alg": "RS256"})
    return private_key, jwk


def test_parse_bearer_token() -> None:
    assert parse_bearer_token("Bearer abc") == "abc"
    assert parse_bearer_token("bearer  abc") == "abc"
    assert parse_bearer_token("Basic abc") is None
    assert parse_bearer_token(None) is None


def test_verifier_returns_identity_and_context() -> None:
    secret = "test-secret-key-0123456789abcdef"
    verifier = TokenVerifier(
        AuthConfig(jwt_secret=secret, jwt_issuer="issuer", jwt_audience="audience")
    )
    token = _token(
        secret,
        sub="user-1",
        tenant_id="tenant-1",
        kb_id="kb-1",
        iss="issuer",
        aud="audience",
    )
    identity = verifier.identity(token)
    assert isinstance(identity, IdentityClaims)
    assert identity.sub == "user-1"
    assert identity.tenant_id == "tenant-1"
    ctx = identity.to_context()
    assert ctx.user_id == "user-1"
    assert ctx.tenant_id == "tenant-1"
    assert ctx.kb_id == "kb-1"


def test_verifier_rejects_expired_token() -> None:
    secret = "test-secret-key-0123456789abcdef"
    verifier = TokenVerifier(AuthConfig(jwt_secret=secret))
    token = jwt.encode(
        {"sub": "u", "exp": int(time.time()) - 10},
        secret,
        algorithm="HS256",
    )
    with pytest.raises(AuthError):
        verifier.verify(token)


def test_verifier_fails_closed_without_key() -> None:
    verifier = TokenVerifier(AuthConfig(jwt_secret=""))
    with pytest.raises(AuthError):
        verifier.verify("not-a-real-jwt")


def test_verifier_follows_jwks_after_key_rotation() -> None:
    """JWKS 模式下按 kid 验签，且 IdP 轮换密钥后无需重建 verifier。"""
    kid_1 = "key-1"
    kid_2 = "key-2"
    private_1, jwk_1 = _rsa_signing_key(kid_1)
    private_2, jwk_2 = _rsa_signing_key(kid_2)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _JwksHandler)
    server.jwks = [jwk_1]
    server.jwks_lock = threading.Lock()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/jwks"
        verifier = TokenVerifier(
            AuthConfig(
                jwt_jwks_url=url,
                jwt_jwks_lifespan=3600,
                jwt_algorithms=("RS256",),
            )
        )
        first_token = jwt.encode(
            {"sub": "user-1", "exp": int(time.time()) + 3600},
            private_1,
            algorithm="RS256",
            headers={"kid": kid_1},
        )
        assert verifier.verify(first_token)["sub"] == "user-1"

        # 模拟 IdP 完成轮换：JWKS 只保留新公钥，token 使用新私钥签名。
        server.jwks = [jwk_2]
        rotated_token = jwt.encode(
            {"sub": "user-2", "exp": int(time.time()) + 3600},
            private_2,
            algorithm="RS256",
            headers={"kid": kid_2},
        )
        assert verifier.verify(rotated_token)["sub"] == "user-2"

        # 旧公钥已被移出 JWKS，旧 token 应被拒绝，而不是沿用缓存密钥。
        with pytest.raises(AuthError, match="Invalid token"):
            verifier.verify(first_token)
    finally:
        server.shutdown()
        server.server_close()
