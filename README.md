# SEAnergy

계명대학교 미래자동차공학전공 '글로벌공학설계프로젝트' — 머신러닝 기반 OBD2 연동 배터리 정밀 진단 및 지능형 안전 관제 플랫폼 (전기차 배터리 SOH 진단)

이번 학기 범위: 데이터 분석 · AI 모델 · API · 웹사이트 파이프라인 구현
(실차 OBD2 연동은 겨울 Bellevue College 공동 글로벌 프로젝트로 이월)

프로젝트 진행 상황, 데이터셋 상세, 회의록 등은 [노션 워크스페이스](https://www.notion.so)에서 관리합니다.

## 데이터셋

용량 문제로 원본 데이터는 이 저장소에 포함하지 않습니다. `preprocessing/` 스크립트로 재생성하세요.

| 데이터셋 | 화학종 | 상태 |
|---|---|---|
| NASA PCoE | LCO | 채택 |
| BatICM | NCM | 채택 · 메인 |
| SNL NMC | NMC | 검토 중 |

## 구조

```
preprocessing/   # 데이터셋별 전처리 스크립트
datasets/        # (git 제외) 원본/전처리 데이터 — preprocessing/ 스크립트로 생성
```
