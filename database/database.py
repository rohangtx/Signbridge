import sqlite3
def init_db(p):
 p.parent.mkdir(parents=True,exist_ok=True); c=sqlite3.connect(p); c.execute('CREATE TABLE IF NOT EXISTS predictions(id INTEGER PRIMARY KEY,mode TEXT,prediction TEXT,confidence REAL,language TEXT,created_at TEXT DEFAULT CURRENT_TIMESTAMP)'); c.commit(); c.close()
def add_prediction(p,mode,pred,conf,lang):
 init_db(p); c=sqlite3.connect(p); c.execute('INSERT INTO predictions(mode,prediction,confidence,language) VALUES(?,?,?,?)',(mode,pred,conf,lang)); c.commit(); c.close()
def recent(p,n=10):
 init_db(p); c=sqlite3.connect(p); r=c.execute('SELECT mode,prediction,confidence,language,created_at FROM predictions ORDER BY id DESC LIMIT ?', (n,)).fetchall(); c.close(); return r
