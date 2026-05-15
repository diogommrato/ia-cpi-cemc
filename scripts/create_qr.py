import sys
from pathlib import Path
import qrcode

if len(sys.argv) != 2:
    print("Uso: python scripts/create_qr.py https://o-teu-link.streamlit.app")
    raise SystemExit(1)

url = sys.argv[1]
out = Path("qr_assistente_tfc.png")
img = qrcode.make(url)
img.save(out)
print(f"QR code criado: {out.resolve()}")
