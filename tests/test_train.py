from src.train import load_config, train_one_seed


def test_load_config_reads_expected_keys():
    config = load_config()

    assert config["seeds"] == [0, 1, 2]
    assert config["total_timesteps"] == 150000
    assert "sac" in config
    assert config["sac"]["gamma"] == 0.99


def test_train_one_seed_smoke_run_produces_model_and_log(tmp_path, monkeypatch):
    import src.train as train_module

    monkeypatch.setattr(train_module, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(train_module, "LOGS_DIR", tmp_path / "logs")

    config = load_config()
    config["total_timesteps"] = 200  # tiny budget: this test only checks wiring, not learning quality

    model_path = train_one_seed(seed=0, config=config)

    assert model_path == tmp_path / "models" / "sac_seed0.zip"
    assert model_path.exists()
    assert (tmp_path / "logs" / "seed0.monitor.csv").exists()
