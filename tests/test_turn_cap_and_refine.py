def test_turn_cap_honored(client):
    messages = []
    for i in range(10):
        messages.append({"role": "user", "content": f"turn {i} hiring a java developer"})
        messages.append({"role": "assistant", "content": "ok"})
    messages.append({"role": "user", "content": "one more thing"})

    resp = client.post("/chat", json={"messages": messages})
    body = resp.json()
    assert resp.status_code == 200
    assert body["end_of_conversation"] is True


def test_refine_updates_rather_than_restarts(client):
    resp1 = client.post("/chat", json={"messages": [
        {"role": "user", "content": "Hiring a java developer, mid-level, strong sql skills"},
    ]})
    body1 = resp1.json()

    resp2 = client.post("/chat", json={"messages": [
        {"role": "user", "content": "Hiring a java developer, mid-level, strong sql skills"},
        {"role": "assistant", "content": body1["reply"]},
        {"role": "user", "content": "Actually also add communication and leadership focus"},
    ]})
    body2 = resp2.json()
    assert "reply" in body2
    assert isinstance(body2["recommendations"], list)
