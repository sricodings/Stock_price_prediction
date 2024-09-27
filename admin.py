import sqlite3
import hashlib

conn = sqlite3.connect('app.db')
c = conn.cursor()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

admin_username = 'admin'
admin_password = 'adminpass' 

try:
    c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", 
              (admin_username, hash_password(admin_password), 'admin'))
    conn.commit()
    print("Admin user created successfully.")
except sqlite3.IntegrityError:
    print("Admin user already exists.")
finally:
    conn.close()