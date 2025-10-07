import pytest
from app.app import app  # Importa la app Flask desde app/app.py

@pytest.fixture
def client():
    # Crea un cliente de pruebas para Flask
    with app.test_client() as client:
        yield client

def test_home_route(client):
    """Prueba que la ruta principal responda correctamente"""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Species Whisper" in response.data  # Ajusta según tu HTML o texto esperado
