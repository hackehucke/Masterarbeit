import json, sys
sys.stdout.reconfigure(encoding='utf-8')
with open('curve_sequence_forecasting.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 7: decoder_embed aus Trainings-Klasse entfernen
src = ''.join(nb['cells'][7]['source'])
src = src.replace(
    "        self.decoder_embed = CurveEncoder(seq_len, d_model)  # fuer Teacher Forcing\n",
    ""
)
nb['cells'][7]['source'] = src

# Cell 15: decoder_embed aus Inference-Klasse entfernen (war dort eh nicht drin, aber sicherheitshalber)
src15 = ''.join(nb['cells'][15]['source'])
src15 = src15.replace(
    "        self.decoder_embed = CurveEncoder(seq_len, d_model)  # fuer Teacher Forcing\n",
    ""
)
nb['cells'][15]['source'] = src15

with open('curve_sequence_forecasting.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Fertig.')
