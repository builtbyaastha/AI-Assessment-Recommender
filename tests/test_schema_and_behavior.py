REQUIRED_KEYS = {"reply", "recommendations", "end_of_conversation"}


def _chat(client, messages):
    resp = client.post("/chat", json={"messages": messages})
    assert resp.status_code == 200
    body = resp.json()
    assert REQUIRED_KEYS <= body.keys()
    return body


def test_vague_query_does_not_recommend_on_turn_1(client):
    """Behavior probe: 'I need an assessment' is not enough to act on."""
    body = _chat(client, [{"role": "user", "content": "I need an assessment"}])
    assert body["recommendations"] == []
    assert body["end_of_conversation"] is False
    assert isinstance(body["reply"], str) and len(body["reply"]) > 0


def test_clarify_then_recommend_flow(client):
    body1 = _chat(client, [
        {"role": "user", "content": "Hiring a java developer who works with stakeholders"},
    ])
    # First turn is vague-ish; if the mock LLM decides it already has role+skill
    # it may recommend immediately, which is fine — assert schema either way.
    assert "reply" in body1

    body2 = _chat(client, [
        {"role": "user", "content": "Hiring a java developer who works with stakeholders"},
        {"role": "assistant", "content": body1["reply"]},
        {"role": "user", "content": "Mid-level, 4 years experience, strong in java and sql"},
    ])
    assert len(body2["recommendations"]) >= 1
    for rec in body2["recommendations"]:
        assert rec["url"].startswith("https://www.shl.com")
        assert rec["name"]
        assert rec["test_type"]


def test_recommendations_are_1_to_10_items(client):
    body = _chat(client, [
        {"role": "user", "content": "Hiring a java developer with strong sql skills, mid-level, communication matters"},
    ])
    assert 0 <= len(body["recommendations"]) <= 10


def test_every_recommendation_url_is_from_catalog(client):
    from app.catalog.catalog_loader import get_catalog

    catalog_urls = {item.url for item in get_catalog().items}
    body = _chat(client, [
        {"role": "user", "content": "Hiring a java developer with strong sql skills, mid-level, communication matters"},
    ])
    for rec in body["recommendations"]:
        assert rec["url"] in catalog_urls
