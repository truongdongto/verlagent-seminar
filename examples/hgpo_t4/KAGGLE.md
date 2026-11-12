# Tái hiện HGPO trên WebShop với Kaggle 2×T4

Thí nghiệm: Qwen2.5-0.5B-Instruct, full fine-tune, so sánh GRPO, GiGPO và HGPO trên WebShop (ALFWorld làm sau nếu còn quota).

**So với paper:**

- **Giữ nguyên:** 8 rollout mỗi nhóm, K = 2, reward, γ = 0,95, α = 1, `mode=mean_norm`, lr 1e-6, KL 0,01, temperature 1,0 khi train và 0,4 khi đánh giá.
- **Thu nhỏ:**
  - 8 task mỗi iteration (paper: 16);
  - **60 iteration** (paper: 160);
  - WebShop tối đa **15 bước**, giống Table 7 của paper (Table 1 dùng 30);
  - mô hình 0.5B (paper: 1.5B).

## 0. Nguyên tắc khi dùng Kaggle


| Thư mục                           | Dùng cho                          | Lý do                                                        |
| --------------------------------- | --------------------------------- | ------------------------------------------------------------ |
| `/tmp/verl-agent`, `/tmp/venv310` | Code và venv, cài lại mỗi session | Không tốn dung lượng output 20 GB                            |
| `/kaggle/working/checkpoints`     | Checkpoint và log                 | Được lưu khi commit version, nên session sau train tiếp được |


**Cách chạy:**

- Mỗi lần chạy là một **"Save Version → Save & Run All (Commit)"**, với Accelerator **GPU T4 ×2** và **Internet bật**.
- Một version được chạy tối đa 12 giờ. `run_queue.sh` tự dừng trước giới hạn đó. Version sau gắn output của version trước làm input và train tiếp từ checkpoint mới nhất.
- Checkpoint của 0.5B full fine-tune khoảng 6–7 GB, và chỉ giữ checkpoint mới nhất (`max_actor_ckpt_to_keep=1`).

**Trước khi bắt đầu:** push code mới nhất lên repo GitHub của bạn (`verlagent-seminar`), vì notebook sẽ clone từ đó.

## 1. Notebook: các cell cài đặt (chạy lại mỗi session, khoảng 20–30 phút)

**Cell 1: biến môi trường.** Nếu dùng W&B, thêm secret `WANDB_API_KEY` trong menu Add-ons → Secrets.

```python
import os
os.environ["REPO"] = "/tmp/verl-agent"
os.environ["VENV"] = "/tmp/venv310"
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
except Exception:
    pass  # không có W&B: chỉ log ra console và file
```

**Cell 2: clone code, Python 3.10, Java 11** (WebShop cần Python ≤ 3.10; pyserini cần Java 11)

- Image Kaggle hiện là **Ubuntu 24.04 (noble)**: apt không còn gói `python3.10`, nên Python 3.10 được cài bằng `uv` (bản build độc lập, đã kèm header để build package).
- Dòng `W: Skipping acquire of configured file 'main/source/Sources' ... r2u.stat.illinois.edu` chỉ là cảnh báo của repo R có sẵn trong image, bỏ qua được.

```bash
%%bash
set -e
rm -rf $REPO && git clone --depth 1 https://github.com/truongdongto/verlagent-seminar.git $REPO
# Java 11 cho pyserini (có trong apt của cả Ubuntu 22.04 và 24.04)
sudo apt-get update -qq 2>&1 | grep -v "^W:" || true
sudo apt-get install -y -qq openjdk-11-jdk-headless > /dev/null
/usr/lib/jvm/java-11-openjdk-amd64/bin/java -version
# Python 3.10 qua uv (cài vào ~/.local/share/uv, không tốn dung lượng output)
python3 -m pip install -q uv
python3 -m uv python install 3.10
rm -rf $VENV && python3 -m uv venv --seed --python 3.10 $VENV
$VENV/bin/python -m pip install -q --upgrade pip setuptools wheel
$VENV/bin/python --version
```

**Cell 3: thư viện.**

- Torch được ghim ở 2.6.0, vì vLLM 0.8.5 cần torch 2.6.0 và WebShop cũng ghim bản này.
- **Không cài flash-attn**, vì T4 không chạy được FlashAttention 2.
- `numpy==1.26.4` cũng được ghim: spacy 3.7.2 (WebShop) được build cho numpy 1.x, trong khi `opencv-python-headless` mới nhất (phụ thuộc của vLLM) sẽ kéo numpy lên 2.x nếu không ghim.
- `pyserini==0.17.0` được cài `--no-deps` để bỏ `nmslib`: nmslib không có wheel cho Python 3.10 và thường build lỗi. WebShop chỉ dùng `LuceneSearcher`, không import nmslib.
- Mỗi bước ghi log riêng vào `/kaggle/working/install_logs/`. Nếu một bước lỗi, cell in ra các dòng lỗi của bước đó rồi dừng.

```bash
%%bash
source $VENV/bin/activate
cd $REPO
LOG=/kaggle/working/install_logs; mkdir -p $LOG
printf "torch==2.6.0\ntransformers==4.51.1\nnumpy==1.26.4\n" > /tmp/constraints.txt
grep -v "^pyserini" agent_system/environments/env_package/webshop/webshop/requirements.txt > /tmp/webshop_req.txt
step() {
  name=$1; shift
  echo "[$(date +%T)] $name"
  if ! "$@" > $LOG/$name.log 2>&1; then
    echo "===== FAIL at $name (full log: $LOG/$name.log) ====="
    grep -iE "error|conflict|incompatible|requires" $LOG/$name.log | tail -n 25
    echo "----- last lines -----"; tail -n 20 $LOG/$name.log
    exit 1
  fi
}
step 1_torch    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
step 2_webshop  pip install -r /tmp/webshop_req.txt -c /tmp/constraints.txt
step 3_pyserini pip install --no-deps pyserini==0.17.0
step 4_pyserini_deps pip install "Cython>=0.29.21" "pyjnius>=1.4.0" sentencepiece onnxruntime lightgbm faiss-cpu -c /tmp/constraints.txt
step 5_vllm     pip install vllm==0.8.5 -c /tmp/constraints.txt
step 6_verl     pip install -e . -c /tmp/constraints.txt
step 7_spacy_sm python -m spacy download en_core_web_sm   # WebShop chỉ dùng model sm (goal.py); không cần en_core_web_lg
python -c "import torch, vllm, transformers, tensordict, numpy; print(torch.__version__, vllm.__version__, transformers.__version__, tensordict.__version__, numpy.__version__)"
JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 python -c "from pyserini.search.lucene import LuceneSearcher; import spacy; spacy.load('en_core_web_sm'); print('webshop deps OK')"
```

Nếu một bước FAIL, gửi mình phần cell in ra (từ dòng `===== FAIL at ...`). **Đừng nới ràng buộc torch hay numpy**.

**Cell 4: dữ liệu WebShop.**

- WebShop trong verl-agent dùng `use_small=True`: 1.000 sản phẩm, và index tìm kiếm cũng được dựng từ 1.000 sản phẩm này. Vì vậy chỉ cần bản `small`.
- **Lần đầu:** tải về và dựng index, khoảng 5–10 phút.
- **Các lần sau:** lưu thư mục này thành Kaggle Dataset `webshop-small` rồi copy vào (xem dòng `elif`).

- Giống Cell 3, mỗi bước ghi log vào `/kaggle/working/install_logs/` và in dòng lỗi nếu hỏng.
- Chỉ dựng index `indexes` (1.000 sản phẩm, đúng index mà verl-agent dùng); bỏ 3 index phụ `indexes_100/1k/100k` của `run_indexing.sh`.
```bash
%%bash
source $VENV/bin/activate
python -c "import tqdm, pyserini, spacy" 2>/dev/null || { echo "venv $VENV is missing the Cell 3 packages: run Cell 2 and Cell 3 first"; exit 1; }
W=$REPO/agent_system/environments/env_package/webshop/webshop
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
LOG=/kaggle/working/install_logs; mkdir -p $LOG
step() {
  name=$1; shift
  echo "[$(date +%T)] $name"
  if ! "$@" > $LOG/$name.log 2>&1; then
    echo "===== FAIL at $name (full log: $LOG/$name.log) ====="
    tail -n 30 $LOG/$name.log
    exit 1
  fi
}
if [ -d /kaggle/input/webshop-small/data ]; then
  echo "copy WebShop data from the Kaggle Dataset webshop-small"
  cp -r /kaggle/input/webshop-small/data $W/
  mkdir -p $W/search_engine && cp -r /kaggle/input/webshop-small/indexes $W/search_engine/
else
  mkdir -p $W/data && cd $W/data
  # Google Drive (setup.sh gốc) chặn tải từ Kaggle ("many accesses") -> tải từ mirror trên Hugging Face.
  # 3 mirror độc lập (YWZBrandon/webshop-data, HongbangYuan/webshop, Skyler215/VERL_WEBSHOP) có cùng git hash.
  HF=https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/main
  for f in items_shuffle_1000.json items_ins_v2_1000.json items_human_ins.json; do
    step dl_$f curl -fL --retry 3 -o $f $HF/$f
  done
  python - <<'EOF'
import hashlib, sys
expected = {"items_shuffle_1000.json": "08dfda9a1c896e0f69d50e3437dd3fe94419b7d7",
            "items_ins_v2_1000.json": "22837ad99f3173712fc9cb1cce7de6b75fe90595",
            "items_human_ins.json": "b2ab7479b9798e0b62ea394e065f3d506875f090"}
for name, oid in expected.items():
    data = open(name, "rb").read()
    got = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()  # git blob hash, as shown on Hugging Face
    print(f"{name}: {len(data)} bytes, {'OK' if got == oid else 'HASH MISMATCH ' + got}")
    if got != oid: sys.exit(1)
EOF
  [ $? -eq 0 ] || exit 1
  cd $W/search_engine
  mkdir -p resources resources_100 resources_1k resources_100k
  step i1_convert python convert_product_file_format.py
  step i2_index python -m pyserini.index.lucene --collection JsonCollection --input resources --index indexes \
      --generator DefaultLuceneDocumentGenerator --threads 1 --storePositions --storeDocvectors --storeRaw
  # keep a copy to publish as the Kaggle Dataset "webshop-small"
  mkdir -p /kaggle/working/webshop-small && cp -r $W/data $W/search_engine/indexes /kaggle/working/webshop-small/
fi
ls -la $W/data $W/search_engine/indexes
cd $W && python -c "
from pyserini.search.lucene import LuceneSearcher
s = LuceneSearcher('search_engine/indexes'); print('docs in index:', s.num_docs)
print([h.docid for h in s.search('men dress shirt', k=3)])"
```

Nếu `gdown` bị Google Drive chặn, hãy tải 3 file đó về máy, upload thành Dataset, rồi copy vào `$W/data`.

**Cell 5: khôi phục checkpoint của version trước.** Từ version thứ hai trở đi, vào **Add Input → Your Work → Notebook Output Files** và chọn output của version trước. Cell này tự tìm thư mục `checkpoints` trong mọi input.

```bash
%%bash
prev=$(ls -d /kaggle/input/*/checkpoints /kaggle/input/*/*/checkpoints 2>/dev/null | head -1)
if [ -n "$prev" ] && [ ! -d /kaggle/working/checkpoints ]; then
  echo "restore from $prev"; cp -r "$prev" /kaggle/working/
fi
ls /kaggle/working/checkpoints/logs 2>/dev/null || echo "fresh start"
du -sh /kaggle/working/checkpoints 2>/dev/null || true
```



## 2. Lần chạy đầu tiên: preflight và smoke test (khoảng 1 giờ GPU)

```bash
%%bash
source $VENV/bin/activate && cd $REPO
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
python -m pytest tests/fintrade -q -rs
python examples/hgpo_t4/preflight.py all 2>&1 | grep -E "^\[|^---|PREFLIGHT"
```

**Mong đợi:**


| Mục           | Kết quả mong đợi                                                                                  |
| ------------- | ------------------------------------------------------------------------------------------------- |
| pytest        | 27 passed, không còn test bị skip (có 2 test so khớp với `core_hgpo.py` gốc)                      |
| `env`         | 2 GPU T4 sm75; WebShop `data/*` và `indexes` đều PASS                                             |
| `vllm`        | PASS cả phần sinh văn bản fp16 lẫn sleep/wake_up                                                  |
| `grad`        | Một dòng khuyến nghị `TRAIN_DTYPE=fp16` hoặc `fp32`. **Dùng đúng giá trị này cho mọi run**        |
| `webshop-mem` | PASS cho cả dòng "training" và "final eval". Mỗi Ray actor chứa `ENVS_PER_WORKER=16` env; các env cùng nhóm dùng chung một WebShop server (1 env ≈ 1,2 GB RAM, nên 64 actor riêng sẽ không vừa 33 GB). Ngưỡng PASS là 40% RAM vì 2 GPU worker, TaskRunner và Ray cần thêm ~14 GB. Nếu WARN/FAIL, thêm `os.environ["JAVA_TOOL_OPTIONS"]="-Xmx256m"` ở Cell 1 rồi chạy lại |


Smoke test: 2 iteration cho mỗi thuật toán, với 2 task × 8 rollout.

```bash
%%bash
source $VENV/bin/activate && cd $REPO
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
DTYPE=fp16             # fp32 if `preflight grad` does not PASS
ALGOS="grpo gigpo hgpo"
mkdir -p /kaggle/working/smoke
for a in $ALGOS; do
  # RAM monitor: every 10 s, write used/total RAM and the 6 largest processes
  ( while sleep 10; do
      free -m | awk -v t=$(date +%T) '/Mem/{printf "%s used %.1f / %.1f GB\n", t, $3/1024, $2/1024}'
      ps -eo rss=,args= --sort=-rss | head -6 | awk '{rss=$1; $1=""; printf "   %5.1f GB %s\n", rss/1048576, substr($0, 2, 100)}'
    done ) > /kaggle/working/smoke/mem_$a.log 2>&1 &
  MON=$!
  SMOKE=1 TRAIN_DTYPE=$DTYPE bash examples/hgpo_t4/train.sh webshop $a > /kaggle/working/smoke/$a.log 2>&1
  rc=$?; kill $MON
  echo "== $a exit=$rc | peak RAM: $(grep -oE 'used [0-9.]+' /kaggle/working/smoke/mem_$a.log | sort -k2 -n | tail -1) GB"
  grep -E "Traceback|Error|OutOfMemory|nan" /kaggle/working/smoke/$a.log | grep -v "UserWarning" | tail -3
  grep -E "step:[0-9]+ - " /kaggle/working/smoke/$a.log | tail -1 | grep -oE "(timing_s/(step|gen|old_log_prob|ref|update_actor|testing)|prompt_length/(mean|max)|actor/loss_scale|val/success_rate|episode/valid_action_ratio|actor/pg_loss|actor/grad_norm|context_groups/[0-9]-context/utilization|adv_bias/oracleK/[a-z0-9_]+|response_length/clip_ratio|perf/max_memory_allocated_gb):[-0-9.e]+"
  if [ $rc -ne 0 ]; then  # stop at the first failure: the other algorithms would fail the same way
    echo "--- last RAM samples before the failure"; tail -14 /kaggle/working/smoke/mem_$a.log
    echo "--- Ray OOM report"; grep -i -A14 "memory consumers" /kaggle/working/smoke/$a.log | head -18
    break
  fi
done
true  # keep the cell from raising CalledProcessError when the last grep finds nothing
```

`DTYPE`: fp16 thô (không loss scaling) cho cosine(fp16, fp32)=0,97 và 27% gradient bị underflow về 0. `dp_actor` giờ tự bật dynamic loss scaling khi `TRAIN_DTYPE=fp16` (log ra `actor/loss_scale`). Dùng fp16 nếu `preflight grad` báo PASS (trên Kaggle T4: PASS, fp16 nhanh hơn fp32 4,9 lần), ngược lại dùng fp32. vLLM luôn sinh văn bản bằng fp16.

**Gửi mình:** output của preflight, output của smoke test, và giá trị `timing_s/step`. Từ `timing_s/step` mình sẽ ước lượng số session cần cho 60 iteration.

## 3. Grid chính: GRPO, GiGPO, HGPO trên WebShop

Mỗi version chạy Cell 1–5 rồi đến cell dưới đây. Lặp lại cho đến khi cả 3 run có file `.done`.

```bash
%%bash
source $VENV/bin/activate && cd $REPO
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
SESSION_BUDGET_MIN=630 TRAIN_DTYPE=fp16 EPOCHS=60 bash examples/hgpo_t4/run_queue.sh webshop
tail -n 3 /kaggle/working/checkpoints/logs/*.log
```

- `SESSION_BUDGET_MIN=630` (10,5 giờ) chừa thời gian cho phần cài đặt và lưu output.
- Các run chạy lần lượt theo thứ tự GRPO, GiGPO, HGPO. Run nào chưa xong sẽ train tiếp ở version sau. Bạn mất tối đa `SAVE_FREQ=5` iteration chưa kịp lưu.
- Nếu một run báo `FAIL`, hàng đợi sẽ dừng để không tốn quota. Khi đó gửi mình đoạn cuối của `logs/webshop_<algo>.log`.



## 4. Đánh giá cuối và tổng hợp kết quả

Đánh giá theo paper: checkpoint cuối, 3 seed đánh giá (123, 456, 789), 128 task WebShop mỗi seed.

```bash
%%bash
source $VENV/bin/activate && cd $REPO
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
TRAIN_DTYPE=fp16 bash examples/hgpo_t4/eval_all.sh webshop
pip install -q matplotlib
python examples/hgpo_t4/summarize.py --log_dir /kaggle/working/checkpoints/logs --out /kaggle/working/results
```

**Kết quả** nằm trong `/kaggle/working/results/`:


| File                         | Tương ứng trong paper                                                |
| ---------------------------- | -------------------------------------------------------------------- |
| `table1_webshop.csv`         | Table 1: success rate và task score, ×100, mean ± std                |
| `table2_webshop.csv`         | Table 2: step utilization và kích thước nhóm theo từng mức k-context |
| `fig2_webshop.png`           | Fig. 2: độ lệch advantage so với oracle theo iteration               |
| `metrics_webshop_<algo>.csv` | Toàn bộ metric theo từng iteration (training dynamics, Fig. 8)       |




## 5. ALFWorld (làm sau, nếu còn quota)

- Thêm vào Cell 3: `pip install gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld`, rồi chạy `alfworld-download -f`. Nên lưu `~/.cache/alfworld` thành Dataset.
- Chạy: `ALGOS="gigpo hgpo grpo" bash examples/hgpo_t4/run_queue.sh alfworld`.
- Đánh giá: `bash examples/hgpo_t4/eval_all.sh alfworld`. Script tự đánh giá cả in- và out-of-distribution.

