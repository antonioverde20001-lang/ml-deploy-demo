# File vuoto: la sua sola presenza alla radice fa sì che pytest inserisca
# la root del progetto in sys.path, così tests/test_api.py può fare
# `from app import app` senza bisogno di un pacchetto installato.
