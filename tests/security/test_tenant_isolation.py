"""
Security & Multi-Tenant Isolation Proofs.
Proves zero cross-tenant leakage, ABAC clearance enforcement, and token integrity.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.security import create_access_token
from app.schemas.rag import ClearanceLevel, DocumentIngestRequest, QueryRequest
from app.services.ingestion.pipeline import ingestion_service
from app.services.retrieval.engine import hybrid_retriever


client = TestClient(app)


def test_proof_1_zero_cross_tenant_leakage():
    """
    SECURITY PROOF 1:
    Tenant A ingests a secret document.
    Tenant B searches for the exact secret keyword.
    Invariant: Zero chunks from Tenant A must ever be returned to Tenant B.
    """
    # Tenant A (Acme Corp) ingests secret
    token_a = create_access_token(
        subject="alice",
        tenant_id="tenant_acme",
        roles=["engineering"],
        clearance=ClearanceLevel.INTERNAL
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

    # Tenant B (Cyberdyne Corp) queries for the exact secret
    token_b = create_access_token(
        subject="mallory",
        tenant_id="tenant_cyberdyne",
        roles=["engineering"],
        clearance=ClearanceLevel.INTERNAL
    )
    headers_b = {"Authorization": f"Bearer {token_b}"}

    query_b = {"query": "What is the confidential Apollo encryption master key?", "top_k": 5}
    res_query = client.post("/api/v1/rag/query", json=query_b, headers=headers_b)
    
    assert res_query.status_code == 200
    data = res_query.json()
    
    # HARD INVARIANT: Zero citations, status must be INSUFFICIENT_EVIDENCE
    assert data["status"] == "INSUFFICIENT_EVIDENCE"
    assert len(data["citations"]) == 0
    assert data["tenant_id"] == "tenant_cyberdyne"


def test_proof_2_abac_clearance_enforcement():
    """
    SECURITY PROOF 2:
    An executive document is indexed under CONFIDENTIAL classification.
    An engineer with INTERNAL clearance searches for executive compensation.
    Invariant: The confidential document must be mathematically excluded from retrieval.
    """
    token_admin = create_access_token(
        subject="hr_admin",
        tenant_id="tenant_globex",
        roles=["hr", "executive"],
        clearance=ClearanceLevel.CONFIDENTIAL
    )
    headers_admin = {"Authorization": f"Bearer {token_admin}"}

    doc_confidential = {
        "document_id": "exec_compensation",
        "title": "Executive Compensation 2026",
        "text": "The CEO annual bonus target is set at 2.5 million dollars based on EBITDA.",
        "allowed_roles": ["executive"],
        "classification": "confidential"
    }
    client.post("/api/v1/rag/ingest", json=doc_confidential, headers=headers_admin)

    # Regular engineer with INTERNAL clearance queries for bonus
    token_eng = create_access_token(
        subject="engineer_dan",
        tenant_id="tenant_globex",
        roles=["engineering"],
        clearance=ClearanceLevel.INTERNAL  # Clearance < CONFIDENTIAL
    )
    headers_eng = {"Authorization": f"Bearer {token_eng}"}

    query_payload = {"query": "What is the CEO annual bonus target?", "top_k": 3}
    res = client.post("/api/v1/rag/query", json=query_payload, headers=headers_eng)
    
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "INSUFFICIENT_EVIDENCE"
    assert len(data["citations"]) == 0


def test_proof_3_rbac_role_restriction():
    """
    SECURITY PROOF 3:
    A document restricted to 'finance' role cannot be retrieved by 'marketing' role.
    """
    token_auth = create_access_token(
        subject="finance_lead",
        tenant_id="tenant_initech",
        roles=["finance"],
        clearance=ClearanceLevel.INTERNAL
    )
    doc_finance = {
        "document_id": "q3_tax_writeoffs",
        "title": "Q3 Tax Strategy",
        "text": "Depreciation allowance for cloud servers totaled 450000 dollars.",
        "allowed_roles": ["finance"],
        "classification": "internal"
    }
    client.post("/api/v1/rag/ingest", json=doc_finance, headers={"Authorization": f"Bearer {token_auth}"})

    # Marketing user queries
    token_mkt = create_access_token(
        subject="marketer_pam",
        tenant_id="tenant_initech",
        roles=["marketing"],  # Missing 'finance' role
        clearance=ClearanceLevel.INTERNAL
    )
    res = client.post(
        "/api/v1/rag/query",
        json={"query": "What was the depreciation allowance for cloud servers?", "top_k": 3},
        headers={"Authorization": f"Bearer {token_mkt}"}
    )
    assert res.status_code == 200
    assert res.json()["status"] == "INSUFFICIENT_EVIDENCE"


def test_proof_4_token_tampering_rejected():
    """
    SECURITY PROOF 4:
    Any token with an altered signature or invalid format is rejected with 401.
    """
    valid_token = create_access_token(subject="user", tenant_id="acme")
    tampered_token = valid_token[:-4] + "fake"
    
    res = client.post(
        "/api/v1/rag/query",
        json={"query": "hello world"},
        headers={"Authorization": f"Bearer {tampered_token}"}
    )
    assert res.status_code == 401
    assert "Could not validate credentials" in res.json()["detail"]


def test_proof_5_unauthenticated_ingress_blocked():
    """
    SECURITY PROOF 5:
    Accessing /query or /ingest without Authorization header is blocked.
    """
    res = client.post("/api/v1/rag/query", json={"query": "unauthorized query"})
    assert res.status_code == 401
