def test_fit(client):
    response = client.post("/fit", json={"texts": []})
    assert response.status_code == 200

    response_json = response.json()
    assert "model_id" in response_json
    assert "version" in response_json and response_json["version"] == 1
    assert "predictions" in response_json
    assert len(response_json["predictions"]["topics"]) == len(
        response_json["predictions"]["probabilities"]
    )
