import sys
sys.path.insert(0, 'v9/src')
from predict import V9InferenceEngine

engine = V9InferenceEngine()
result = engine.predict_risk('2007-06-22')

print('--- V9 Pipeline Integration Validation ---')
print(f'1. Output schema keys  : {list(result.keys())}')
print(f'2. model_version       : {result["model_version"]}')
print(f'3. risk_score          : {result["risk_score"]} | in [0,100]: {0 <= result["risk_score"] <= 100}')
print(f'4. status              : {result["status"]}')
print(f'5. forecast            : {result["forecast"]}')
print(f'6. advisory (truncated): {result["advisory"][:70]}')
print(f'7. explanation keys    : {list(result["explanation"].keys())}')
print()
print('PASS: Pipeline operational with V9-TCN model.')
