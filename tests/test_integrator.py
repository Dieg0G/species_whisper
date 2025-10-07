import sys, os
import pytest

# Asegura que la ruta de app/ esté disponible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app')))

from integrator import AudioAnalyzer, get_species_prediction


@pytest.fixture
def mock_analyzer(monkeypatch):
    """Crea un analizador con métodos simulados para no usar modelo real."""
    analyzer = AudioAnalyzer()
    analyzer.initialize = lambda: True  # siempre inicializa bien
    analyzer.analyze_audio = lambda path, conf=0.1: {
        "success": True,
        "status": "mock test ok",
        "predictions": [("Cardinal rojo", 0.98)],
        "top_prediction": ("Cardinal rojo", 0.98),
        "prediction_count": 1
    }
    return analyzer


def test_get_species_prediction(monkeypatch, mock_analyzer):
    """Prueba básica de la función get_species_prediction simulando el análisis."""
    monkeypatch.setattr("integrator.get_audio_analyzer", lambda: mock_analyzer)

    species = get_species_prediction("fake_audio.wav")

    assert species == "Cardinal rojo"
