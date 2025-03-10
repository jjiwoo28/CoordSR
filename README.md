
### 모델 관련 (modules/)

- **run_CoordX.py**: CoordX 모델 만 학습
- **run_R2L.py**: teacher를 통해 R2L 학습
- **run_CoordSR.py**: CoordX + teacher를 통해 SR 학습
- **run_CoordSR_combine.py**: 추론시간 측정을 위해 CoordX , SR이 결합된체 학습

### 환경
- 기존적으로 wire_env 를 기준으로 추가된 환경을 사용
- env.yml 을 참고
