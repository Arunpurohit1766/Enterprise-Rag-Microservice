"""
Security & Multi-Tenant Isolation Proofs.
Proves zero cross-tenant leakage, ABAC clearance enforcement,
OAuth2 scope boundaries, write policy controls, and token integrity.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.security import create_access_token
from app.schemas.rag import ClearanceLevel


client = TestClient(app)


def test_proof_1_zero_cross_tenant_leakage():
    """Tenant A ingests a secret. Tenant B queries it. Expected: Zero leakage."""
    token_a = create_access_token(
        subject="alice",
        tenant_id="tenant_acme",
        roles=["engineering"],
        clearance=ClearanceLevel.INTERNAL,
        scopes=["knowledge:read", "knowledge:write"]
    )
    headers_a = {"Authorization": f"Bearer {token_a}"}

    doc_a = {
        "document_id": "acme_secret_project",
        "title": "Project Apollo Secrets",
        "text": "The confidential Apollo encryption master key is ALPHA-9928-OMEGA.",
        "allowed_roles": ["engineering"],
        "classification": "internal"
    }
    res_ingest = client.post("/api/v1/rag/ingest", json=doc_a, headers=headers_a)
    assert res_ingest.status_code == 201

    token_b = create_access_token(
        subject="mallory",
        tenant_id="tenant_cyberdyne",
        roles=["engineering"],
        clearance=ClearanceLevel.INTERNAL,
        scopes=["knowledge:read", "knowledge:write"]
    )
    headers_b = {"Authorization": f"Bearer {token_b}"}

    query_b = {"query": "What is the confidential Apollo encryption master key?", "top_k": 5}
    res_query = client.post("/api/v1/rag/query", json=query_b, headers=headers_b)

    assert res_query.status_code == 200
    data = res_query.json()
    assert data["status"] == "INSUFFICIENT_EVIDENCE"
    assert len(data["citations"]) == 0
    assert data["tenant_id"] == "tenant_cyberdyne"


def test_proof_2_abac_clearance_enforcement():
    """Engineer with INTERNAL clearance cannot retrieve CONFIDENTIAL executive document."""
    token_admin = create_access_token(
        subject="hr_admin",
        tenant_id="tenant_globex",
        roles=["hr", "executive", "admin"],
        clearance=ClearanceLevel.CONFIDENTIAL,
        scopes=["knowledge:read", "knowledge:write"]
    )
    headers_admin = {"Authorization": f"Bearer {token_admin}"}

    doc_confidential = {
        "document_id": "exec_compensation",
        "title": "Executive Compensation 2026",
        "text": "The CEO annual bonus target is set at 2.5 million dollars based on EBITDA.",
        "allowed_roles": ["executive"],
        "classification": "confidential"
    }
    res = client.post("/api/v1/rag/ingest", json=doc_confidential, headers=headers_admin)
    assert res.status_code == 201

    token_eng = create_access_token(
        subject="engineer_dan",
        tenant_id="tenant_globex",
        roles=["engineering"],
        clearance=ClearanceLevel.INTERNAL,
        scopes=["knowledge:read", "knowledge:write"]
    )
    headers_eng = {"Authorization": f"Bearer {token_eng}"}

    query_payload = {"query": "What is the CEO annual bonus target?", "top_k": 3}
    res = client.post("/api/v1/rag/query", json=query_payload, headers=headers_eng)

    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "INSUFFICIENT_EVIDENCE"
    assert len(data["citations"]) == 0


def test_proof_3_rbac_role_restriction():
    """Document restricted to 'finance' role cannot be retrieved by 'marketing' role."""
    token_auth = create_access_token(
        subject="finance_lead",
        tenant_id="tenant_initech",
        roles=["finance"],
        clearance=ClearanceLevel.INTERNAL,
        scopes=["knowledge:read", "knowledge:write"]
    )
    doc_finance = {
        "document_id": "q3_tax_writeoffs",
        "title": "Q3 Tax Strategy",
        "text": "Depreciation allowance for cloud servers totaled 450000 dollars.",
        "allowed_roles": ["finance"],
        "classification": "internal"
    }
    client.post("/api/v1/rag/ingest", json=doc_finance, headers={"Authorization": f"Bearer {token_auth}"})

    token_mkt = create_access_token(
        subject="marketer_pam",
        tenant_id="tenant_initech",
        roles=["marketing"],
        clearance=ClearanceLevel.INTERNAL,
        scopes=["knowledge:read", "knowledge:write"]
    )
    res = client.post(
        "/api/v1/rag/query",
        json={"query": "What was the depreciation allowance for cloud servers?", "top_k": 3},
        headers={"Authorization": f"Bearer {token_mkt}"}
    )
    assert res.status_code == 200
    assert res.json()["status"] == "INSUFFICIENT_EVIDENCE"


def test_proof_4_token_tampering_rejected():
    """Any token with an altered signature or invalid format is rejected with 401."""
    valid_token = create_access_token(subject="user", tenant_id="acme")
    tampered_token = valid_token[:-4] + "fake"

    res = client.post(
        "/api/v1/rag/query",
        json={"query": "hello world"},
        headers={"Authorization": f"Bearer {tampered_token}"}
    )
    assert res.status_code == 401


def test_proof_5_unauthenticated_ingress_blocked():
    """Requests lacking Authorization header fail with 401."""
    res = client.post("/api/v1/rag/query", json={"query": "unauthorized query"})
    assert res.status_code == 401


def test_proof_6_scope_enforcement():
    """Principal with only 'knowledge:read' scope cannot call /ingest."""
    read_only_token = create_access_token(
        subject="readonly_user",
        tenant_id="acme",
        scopes=["knowledge:read"]
    )
    headers = {"Authorization": f"Bearer {read_only_token}"}
    doc = {
        "document_id": "attempt",
        "title": "Attempt",
        "text": "Attempting unauthorized write.",
        "allowed_roles": ["engineering"],
        "classification": "internal"
    }
    res = client.post("/api/v1/rag/ingest", json=doc, headers=headers)
    assert res.status_code == 403
    assert "lacks required scope" in res.json()["detail"]


def test_proof_7_write_policy_clearance_elevation_blocked():
    """Principal with INTERNAL clearance cannot ingest a CONFIDENTIAL document."""
    internal_token = create_access_token(
        subject="regular_user",
        tenant_id="acme",
        clearance=ClearanceLevel.INTERNAL,
        scopes=["knowledge:write"]
    )
    headers = {"Authorization": f"Bearer {internal_token}"}
    doc = {
        "document_id": "privilege_elevation",
        "title": "Secret",
        "text": "Secret content.",
        "allowed_roles": ["engineering"],
        "classification": "confidential"
    }
    res = client.post("/api/v1/rag/ingest", json=doc, headers=headers)
    assert res.status_code == 403
    assert "exceeding principal clearance" in res.json()["detail"]
