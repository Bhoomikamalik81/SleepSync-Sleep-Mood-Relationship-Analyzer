"""Automated tests for SleepSync Flask API.

Start API first:
    python app.py

Run tests:
    python test_api.py
"""

import json
import urllib.request
import urllib.error
import unittest

URL = "http://127.0.0.1:5000"

high = {
    "isi_1": 3, "isi_2": 3, "isi_3": 2,
    "isi_4": 3, "isi_5": 2, "isi_6": 3, "isi_7": 3,
    "pss_2": 4, "pss_3": 4, "pss_14": 4, "pss_1": 4,
    "age": 20,
    "gender": "female",
    "smoke": "never smokes",
    "drink": "never drinks"
}

low = dict(
    high,
    isi_1=0, isi_2=0, isi_3=0, isi_4=1,
    isi_5=0, isi_6=0, isi_7=0,
    pss_2=1, pss_3=1, pss_14=1, pss_1=1
)


def call(path, body=None):
    data = json.dumps(body).encode() if body is not None else None

    req = urllib.request.Request(
        URL + path,
        data=data,
        headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, json.loads(response.read())

    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


class TestSleepSyncAPI(unittest.TestCase):

    def test_health(self):
        status, out = call("/health")

        self.assertEqual(status, 200)
        self.assertEqual(out["status"], "ok")
        self.assertIn("model", out)

    def test_questions(self):
        status, out = call("/questions")

        self.assertEqual(status, 200)
        self.assertIn("sleep_questions", out)
        self.assertIn("stress_questions", out)
        self.assertIn("age_range", out)
        self.assertIn("categories", out)
        self.assertIn("risk_levels", out)
        self.assertIn("helpline", out)

        self.assertEqual(
            len(out["sleep_questions"]["columns"]), 7
        )
        self.assertEqual(
            out["sleep_questions"]["answer_range"], [0, 4]
        )
        self.assertEqual(
            out["stress_questions"]["answer_range"], [1, 5]
        )

    def check_prediction(self, body):
        status, out = call("/predict", body)

        self.assertEqual(status, 200)
        self.assertIn("risk_score", out)
        self.assertIn("level", out)
        self.assertIn("title", out)
        self.assertIn("message", out)
        self.assertIn("biggest_sleep_concern", out)
        self.assertIn("disclaimer", out)

        self.assertIsInstance(out["risk_score"], int)
        self.assertGreaterEqual(out["risk_score"], 0)
        self.assertLessEqual(out["risk_score"], 100)

        self.assertIn(
            out["level"],
            ["steady", "disturbed", "needs_attention"]
        )

    def test_high_answers(self):
        self.check_prediction(high)

    def test_low_answers(self):
        self.check_prediction(low)

    def test_missing_field(self):
        data = dict(high)
        del data["isi_3"]

        status, out = call("/predict", data)

        self.assertEqual(status, 400)
        self.assertEqual(out["error"], "Invalid input")

    def test_out_of_range_answer(self):
        data = dict(high, isi_1=9)

        status, out = call("/predict", data)

        self.assertEqual(status, 400)
        self.assertEqual(out["error"], "Invalid input")

    def test_invalid_age(self):
        data = dict(high, age=50)

        status, out = call("/predict", data)

        self.assertEqual(status, 400)
        self.assertEqual(out["error"], "Invalid input")

    def test_invalid_category(self):
        data = dict(high, gender="unknown")

        status, out = call("/predict", data)

        self.assertEqual(status, 400)
        self.assertEqual(out["error"], "Invalid input")

if __name__ == "__main__":
    unittest.main(verbosity=2)