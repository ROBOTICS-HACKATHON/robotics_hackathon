"""Exercise opt-in geometry and the camera HTTP boundary without an arm."""

import argparse
import json
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import see  # noqa: E402
import tipvision  # noqa: E402
from vision_options import Error, VisionOptions, add_vision_arguments, attempt, load_parameters, vision_options  # noqa: E402


class VisionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.original_paths = (see.ScreenCamera.CACHE, tipvision.MODEL_FILE, tipvision.PICK_FILE, tipvision.GEOM_FILE)
        see.ScreenCamera.CACHE = self.root / "screen.json"
        tipvision.MODEL_FILE = self.root / "model.json"
        tipvision.PICK_FILE = self.root / "color.json"
        tipvision.GEOM_FILE = self.root / "geometry.json"
        self.addCleanup(self.restore_paths)
        self.specs = [str(self.root / f"camera{i}.png") for i in range(2)]
        for spec in self.specs:
            cv2.imwrite(spec, np.full((720, 960, 3), 255, np.uint8))
        self.entry = {"image_size": [960, 720], "fx": 780, "fy": 820, "cx": 470, "cy": 350,
                      "distortion": [-0.08, 0.01, 0.002, -0.003, 0.0]}
        self.calibration = self.root / "calibration.json"
        self.write_calibration(self.specs)

    def restore_paths(self):
        see.ScreenCamera.CACHE, tipvision.MODEL_FILE, tipvision.PICK_FILE, tipvision.GEOM_FILE = self.original_paths

    def write_calibration(self, specs):
        self.calibration.write_text(json.dumps({spec: self.entry for spec in specs}))

    def vision(self, **options):
        return tipvision.TipVision(self.specs, options=VisionOptions(**options))

    def projected(self, centre, tip=(150, 100, -50), entry=None):
        entry = entry or self.entry
        K = np.array([[entry["fx"], 0, entry["cx"]], [0, entry["fy"], entry["cy"]], [0, 0, 1]], float)
        C = np.array(centre, float)
        z = np.array([150, 100, 0]) - C
        z /= np.linalg.norm(z)
        x = np.cross([0, 1, 0], z)
        x /= np.linalg.norm(x)
        R = np.stack([x, np.cross(z, x), z])
        rvec = cv2.Rodrigues(R)[0]
        object_points = np.array([[0, 0, 0], [300, 0, 0], [300, 200, 0], [0, 200, 0], tip], float)
        pixels = cv2.projectPoints(object_points, rvec, -R @ C, K, np.array(entry["distortion"]))[0].reshape(-1, 2)
        quad, px = pixels[:4], pixels[4]
        H = cv2.getPerspectiveTransform(quad.astype(np.float32), tipvision.UNIT)
        uv = cv2.perspectiveTransform(px.astype(np.float32).reshape(1, 1, 2), H)[0, 0]
        return {"quad": quad.tolist(), "px": px.tolist(), "shape": [720, 960], "uv": uv.tolist(), "area": 100}

    def views(self, height=50, second_z=-450):
        return [self.projected((-120, -150, -450), (150, 100, -height)),
                self.projected((420, -150, second_z), (150, 100, -height))]

    def test_calibrated_height_with_equal_and_different_camera_heights(self):
        vision = self.vision(calibration=self.calibration, screen_height_mm=200, check_geometry=True)
        for second_z in (-450, -320):
            for height in (0, 20, 50, 100):
                estimate = vision.estimate(self.views(height, second_z))
                self.assertNotIn("rejection", estimate)
                self.assertAlmostEqual(estimate["height_mm"], height, places=3)
                np.testing.assert_allclose(estimate["tip_uv"], [0.5, 0.5], atol=1e-5)
                self.assertLess(estimate["gap_mm"], 0.01)

    def test_rotated_phone_and_calibration_resolution_mismatch(self):
        vision = self.vision(calibration=self.calibration, screen_height_mm=200)
        obs = self.views()
        vision.cams[0].src.rot180 = True
        obs[0]["quad"] = (np.array([959, 719]) - obs[0]["quad"]).tolist()
        obs[0]["px"] = (np.array([959, 719]) - obs[0]["px"]).tolist()
        self.assertAlmostEqual(vision.estimate(obs)["height_mm"], 50, places=3)
        obs[0]["shape"] = [480, 640]
        rejected = vision.estimate(obs)
        self.assertIn("resolution", rejected["rejection"])
        self.assertIsNone(rejected["landing"])

    def test_geometry_rejections_do_not_fall_back_to_uncorrected_2d(self):
        self.entry = dict(self.entry, distortion=[0.0] * 5)
        self.write_calibration(self.specs)
        vision = self.vision(calibration=self.calibration, screen_height_mm=200, check_geometry=True)
        near_parallel = [self.projected((150, 100, -450)), self.projected((151, 100, -450))]
        self.assertIn("angle", vision.estimate(near_parallel)["rejection"])
        self.assertIn("workspace", vision.estimate(self.views(height=400))["rejection"])
        behind = [self.projected((-120, -150, -450), (150, 100, -900)),
                  self.projected((420, -150, -450), (150, 100, -900))]
        self.assertIn("behind", vision.estimate(behind)["rejection"])
        vision.learn(near_parallel, (0.5, 0.5))
        self.assertEqual(vision.data["samples"], [])

    def test_manual_focal_and_screen_dimensions_override_distance_teaching(self):
        tipvision.GEOM_FILE.write_text(json.dumps({"aspect": 1.8, "focal_rel": {spec: 0.5 for spec in self.specs}}))
        vision = self.vision(focal_px=(780, 820), screen_height_mm=200)
        self.assertEqual(vision.aspect, 1.5)
        self.assertEqual(vision._pose(0, self.views()[0])["K"][0, 0], 780)
        self.assertIn("disabled", vision.teach_distance(self.views(), 50)[1])
        vision.forget_distance()
        self.assertEqual(vision.aspect, 1.5)

    def test_validation_cannot_skip_rejected_samples(self):
        vision = self.vision(calibration=self.calibration, screen_height_mm=200, diagnostics=True)
        samples = [{"obs": self.views(height), "landed": [0.5, 0.5]} for height in (10, 20, 30, 40, 50, 60)]
        vision.data["samples"] = samples
        vision._fit()
        self.assertTrue(vision.ready)
        samples[0]["obs"][0]["shape"] = [480, 640]
        vision._fit()
        self.assertFalse(vision.ready)

    def test_setup_keys_preserve_legacy_and_isolate_parameter_changes(self):
        legacy = self.vision()
        self.assertEqual(legacy.key(), "None|" + "|".join(self.specs))
        first = self.vision(setup_id="stand-a", focal_px=(780, 820))
        first.data["samples"].append({"obs": [None, None], "landed": [0.5, 0.5]})
        first.save()
        restored = self.vision(setup_id="stand-a", focal_px=(780, 820))
        self.assertEqual(len(restored.data["samples"]), 1)
        changed = self.vision(setup_id="stand-a", focal_px=(790, 820))
        self.assertNotEqual(changed.key(), restored.key())
        self.assertEqual(changed.data["samples"], [])
        restored.color = "green"
        restored.observe()
        self.assertEqual(restored.data["samples"], [])

    def test_invalid_options_and_calibration_are_reported(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--screen-mm", type=float, default=300)
        add_vision_arguments(parser)
        for flags in (["--vision-focal-px", "nan"], ["--screen-height-mm", "0"],
                      ["--vision-stable-frames", "1"], ["--vision-min-ray-angle-deg", "90"]):
            self.assertIsInstance(vision_options(parser.parse_args(flags)), Error)
        self.calibration.write_text('{"phone": {"image_size": [0, 720]}}')
        self.assertIsInstance(load_parameters(self.calibration), Error)

    def image(self, view):
        board = np.full((600, 900, 3), 255, np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        for number, (x, y) in enumerate(((0, 0), (840, 0), (840, 540), (0, 540))):
            marker = cv2.aruco.generateImageMarker(dictionary, number, 60)
            board[y:y + 60, x:x + 60] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        H = cv2.getPerspectiveTransform(np.array([[0, 0], [899, 0], [899, 599], [0, 599]], np.float32),
                                       np.array(view["quad"], np.float32))
        frame = cv2.warpPerspective(board, H, (960, 720), borderValue=(255, 255, 255))
        cv2.circle(frame, tuple(np.round(view["px"]).astype(int)), 7, (255, 0, 255), -1)
        return frame

    def test_real_detection_paired_http_capture_and_stability(self):
        entry = dict(self.entry, distortion=[0.0] * 5)
        frames = [self.image(self.projected(centre, entry=entry)) for centre in ((-120, -150, -450), (420, -150, -450))]
        jpeg = [cv2.imencode(".jpg", frame)[1].tobytes() for frame in frames]

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                index = int(self.path[1])
                stream = self.path.endswith("/video")
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame" if stream else "image/jpeg")
                self.end_headers()
                for _ in range(100 if stream else 1):
                    data = (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg[index])).encode()
                            + b"\r\n\r\n" + jpeg[index] + b"\r\n") if stream else jpeg[index]
                    result = attempt(lambda: self.wfile.write(data))
                    if isinstance(result, Error):
                        break
                    self.wfile.flush()
                    if stream:
                        time.sleep(0.04)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        for suffix in ("shot.jpg", "video"):
            specs = [f"http://127.0.0.1:{server.server_port}/{i}/{suffix}" for i in range(2)]
            self.entry = entry
            self.write_calibration(specs)
            vision = tipvision.TipVision(specs, options=VisionOptions(calibration=self.calibration,
                                        screen_height_mm=200, check_geometry=True, paired_capture=True,
                                        stable_frames=3, diagnostics=True))
            obs = vision.observe()
            self.assertTrue(all(o and o["stable"] for o in obs))
            estimate = vision.estimate(obs)
            self.assertNotIn("rejection", estimate)
            self.assertAlmostEqual(estimate["height_mm"], 50, delta=3)
            np.testing.assert_allclose(estimate["tip_uv"], [0.5, 0.5], atol=0.015)
            self.assertEqual(vision.diagnostics(estimate)["markers_fresh"], [True, True])
            obs[1]["received_at"] += 1
            self.assertIn("receive", vision.estimate(obs)["rejection"])
            obs[1]["received_at"] = obs[0]["received_at"]
            obs[0]["stable"] = False
            self.assertIn("moving", vision.estimate(obs)["rejection"])


if __name__ == "__main__":
    unittest.main()
