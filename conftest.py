import sys
from pathlib import Path

# Agar `from src...` bisa di-import saat pytest dijalankan dari folder mana pun.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
