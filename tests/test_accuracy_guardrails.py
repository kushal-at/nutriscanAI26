import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.vlm import VLM_OUTPUT_SCHEMA, parse_component_response, parse_recognition_response


class AccuracyGuardrailTests(unittest.TestCase):
    def test_idli_meal_keeps_only_visible_components(self):
        recognition = parse_recognition_response('''{
            "is_food": true,
            "dish": "Idli Breakfast",
            "visible_foods": ["Idli", "Sambar", "Coconut Chutney", "Tomato Chutney"]
        }''')
        measured = parse_component_response('''{
            "is_food": true,
            "dish": "Idli Breakfast",
            "components": [
                {"name": "Idli", "fraction": 0.55, "elevation": "medium"},
                {"name": "Sambar", "fraction": 0.2, "elevation": "low"},
                {"name": "Coconut Chutney", "fraction": 0.15, "elevation": "low"},
                {"name": "Tomato Chutney", "fraction": 0.1, "elevation": "low"}
            ]
        }''', 25)
        self.assertEqual(recognition["visible_foods"][0], "Idli")
        self.assertEqual([item["name"] for item in measured["components"]], recognition["visible_foods"])
        self.assertNotIn("Rice", " ".join(item["name"] for item in measured["components"]))

    def test_structured_schema_cannot_request_hidden_ingredients(self):
        component_properties = VLM_OUTPUT_SCHEMA["properties"]["components"]["items"]["properties"]
        self.assertNotIn("submerged_ingredients", component_properties)
        self.assertNotIn("cluster_id", component_properties)


if __name__ == "__main__":
    unittest.main()
