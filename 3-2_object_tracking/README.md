# line_count

도로 영상에서 차가 선을 지날 때만 센다. 한 프레임의 박스만 세면 같은 차를 여러 번 세기 때문에, 추적 id를 붙인 뒤 선과 만날 때만 더한다.

코랩 노트북이 아니라 로컬 스크립트다. GTX 1080(8GB)에서는 720p와 `RFDETRSmall`을 유지한다. 결과는 `runs/{이름}.mp4`에 저장되고, 화면 아래 막대에 차종별 대수가 나온다. 왼쪽이 IN, 오른쪽이 OUT이다. 화면이 있으면 `--show`로 `cv2.imshow` 창을 연다.

주차 차량이 대부분인 거리 영상은 쓰지 않는다. 아래 예시는 차로를 따라 움직이는 고속도로 장면이다.

| 이름 | 영상 | 장면 |
|---|---|---|
| `vehicles` | `examples/vehicles.mp4` | 기존 쿡북 고속도로. 가로선. |
| `a40` | `examples/a40.mp4` | 보훔 A40. 고가에서 내려다본 양방향 아우토반. 가로선. |
| `a43` | `examples/a43.mp4` | 보훔 A43. 고가에서 내려다본 양방향 아우토반. 가까운 차가 크게 보인다. 가로선. |

```bash
./run.sh line_count/run.py --example vehicles
./run.sh line_count/run.py --example a40
./run.sh line_count/run.py --example a43
./run.sh line_count/run.py --example all
./run.sh line_count/run.py --show
```

영상은 없으면 받는다. 출처는 `examples/SOURCE.txt`에 있다.
