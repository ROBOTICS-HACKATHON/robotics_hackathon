# Robot arm in 3 steps: record → train → run

This setup teaches an SO-101 arm a task by showing it about 50 times.
Everything runs in Docker through one command, `./lr`.

You need **two machines**:

| Machine | Does | Needs |
|---|---|---|
| 💻 **Robot laptop** | recording, running the robot | arms + camera plugged in, Docker |
| 🖥️ **GPU machine** | training (a few hours) | NVIDIA or AMD graphics card, Docker or Podman |

One machine can do both if it has a GPU and the arms are plugged into it.

---

## 0. One-time setup (on each machine)

```bash
git clone <this repo> && cd <this repo>
./lr build        # 5–20 min; detects NVIDIA / AMD / no GPU automatically
./lr login        # paste a Hugging Face WRITE token from huggingface.co/settings/tokens
```

No Docker yet? On Ubuntu or WSL: `sudo apt install docker.io && sudo usermod -aG docker $USER`, then log out and back in.

Settings such as ports, camera and timing are in `lr.conf`, which is created on first run.

---

## 1. 💻 Robot laptop: connect and record

**Windows + WSL only:** install `usbipd` in Windows (`winget install usbipd`). Then, once in an **admin PowerShell**, run `usbipd list`, and run `usbipd bind --busid X-Y` for each arm and the camera. Put those bus IDs in `USB_BUSIDS` in `lr.conf`, with the **follower first, then the leader, then the camera**.

Every time you plug things in:

```bash
./lr attach       # WSL: connects arms + camera. On normal Linux it just lists them
```

First time only:

```bash
./lr cams                      # saves a picture from each camera; pick the normal-looking one
                               # and put its /dev/videoN into CAMERA= in lr.conf
./lr calibrate follower        # follow the prompts
./lr calibrate leader
./lr teleop                    # move the leader, the follower should copy. Ctrl+C to stop
```

Check the camera framing (object, target and gripper all visible), then **don't move the camera again**:

```bash
./lr snap
```

Record:

```bash
./lr test                                        # 2 throwaway episodes
./lr record pen_in_cup 50 "Put the pen in the cup"
```

During each episode, do the task with the **leader** arm.

| Key | Action |
|---|---|
| **→** or `n` | done, next episode |
| **←** or `r` | redo this episode |
| **Esc** or `q` | stop (continue later with `... --resume=true`) |

Between episodes you have 15 s to put the object back. Vary its position a little each time.
When all episodes are done, the dataset uploads to Hugging Face.

---

## 2. 🖥️ GPU machine: train

```bash
./lr gputest                  # should say: GPU available: True
./lr train pen_in_cup         # runs in the background, a few hours
./lr logs pen_in_cup          # watch progress (Ctrl+C stops watching, not training)
```

When it finishes, the model is on Hugging Face as `<you>/pen_in_cup_policy`.

---

## 3. 💻 Robot laptop: let it do the task alone

```bash
./lr attach
./lr run pen_in_cup "Put the pen in the cup"
```

The follower moves **by itself**, so keep hands clear. Same camera position and lighting as during recording.

---

## When something goes wrong

| Problem | Fix |
|---|---|
| `Permission denied: /dev/ttyACM0` | `./lr attach` (it fixes permissions) |
| `No frame` / camera timeout | `./lr attach`, then `./lr cams`; the `/dev/videoN` number may have changed, so update `CAMERA=` in `lr.conf`. Still dropping? Set `FPS=15` |
| Follower doesn't move, leader stiffens | Ports swapped: swap `FOLLOWER_PORT` and `LEADER_PORT` in `lr.conf` |
| `not logged in to Hugging Face` | `./lr login` |
| `GPU available: False` | NVIDIA: install the NVIDIA Container Toolkit. AMD: check `/dev/kfd` exists |
| Asks to calibrate again | IDs in `lr.conf` must match the ones you calibrated with |

Run `./lr help` for all commands.
