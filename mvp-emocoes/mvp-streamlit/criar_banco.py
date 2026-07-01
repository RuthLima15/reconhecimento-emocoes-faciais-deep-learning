import sqlite3

# Cria (ou abre) o banco
conexao = sqlite3.connect("interacoes.db")

cursor = conexao.cursor()

# Cria a tabela
cursor.execute("""
CREATE TABLE IF NOT EXISTS predicoes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    data_hora TEXT,
    emocao_prevista TEXT,
    confianca REAL
)
""")

conexao.commit()
conexao.close()

print("Banco criado com sucesso!")