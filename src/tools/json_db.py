import json
from typing import TypeVar

T = TypeVar("T")


class Db:
    def __init__(self, labels_path, car_info_path):
        with open(labels_path) as f:
            self.labels = json.load(f)
        with open(car_info_path) as f:
            self.car_info = json.load(f)

    def get_owners(self):
        return [{"type": i[0], "name": i[1]} for i in self.labels["owners"]]

    def get_years(self):
        return [{"type": i[0], "name": i[1]} for i in self.labels["year"]]

    def get_regions(self):
        return [{"type": i[0], "name": i[1]} for i in self.labels["region"]]

    def get_marks(self):
        return [{"type": i[0], "name": i[1]} for i in self.labels["mark"]]

    def get_gens(self):
        return [{"type": i[0], "name": i[1]} for i in self.labels["super_gen_name"]]

    def get_model(self):
        return [{"type": i[0], "name": i[1]} for i in self.labels["model"]]

    def get_colors(self):
        return [{"type": i[0], "name": i[1]} for i in self.labels["color"]]

    def get_steering_wheel(self):
        return [{"type": i[0], "name": i[1]} for i in self.labels["steering_wheel"]]

    def get_gear_type(self):
        return [{"type": i[0], "name": i[1]} for i in self.labels["gear_type"]]

    def get_engine(self):
        return [{"type": i[0], "name": i[1]} for i in self.labels["engine"]]

    def get_transmission(self):
        return [{"type": i[0], "name": i[1]} for i in self.labels["transmission"]]

    def get_body_type(self):
        return [{"type": i[0], "name": i[1]} for i in self.labels["body_type_type"]]

    def get_complectation(self):
        return [{"type": i[0], "name": i[1]} for i in self.labels["complectation"]]

    @staticmethod
    def _safe_get(d: dict, keys: list):
        for key in keys:
            d = d.get(key, {})
        return d

    def get_complectations_car(self, mark, model, gen):
        compl = Db._safe_get(self.car_info, [mark, model, gen])["complectation"]
        all_compl = self.get_complectation()
        ret = list(
            map(
                lambda el: ({"name": "Нет", "type": el["type"]} if el["name"] == mark + model else el),
                filter(lambda x: x["type"] in compl, all_compl),
            )
        )
        return ret

    def get_engines_car(self, mark, model, gen):
        eng = Db._safe_get(self.car_info, [mark, model, gen])["engine"]
        all_eng = self.get_engine()
        ret = list(filter(lambda x: x["type"] in eng, all_eng))
        return ret

    def get_transmissions_car(self, mark, model, gen):
        tr = Db._safe_get(self.car_info, [mark, model, gen])["transmission"]
        all_tr = self.get_transmission()
        ret = list(filter(lambda x: x["type"] in tr, all_tr))
        return ret

    def get_body_types_car(self, mark, model, gen):
        bd = Db._safe_get(self.car_info, [mark, model, gen])["body_type_type"]
        all_bd = self.get_body_type()
        ret = list(filter(lambda x: x["type"] in bd, all_bd))
        return ret

    def get_years_car(self, mark, model, gen):
        y = list(map(str, Db._safe_get(self.car_info, [mark, model, gen])["year"]))
        all_y = self.get_years()
        ret = list(filter(lambda x: x["type"] in y, all_y))
        return ret

    def get_super_gen_names_car(self, mark, model):
        gens = Db._safe_get(self.car_info, [mark, model]).keys()
        all_gens = self.get_gens()
        ret = list(
            map(
                lambda el: ({"name": "Нет", "type": el["type"]} if el["name"] == mark + model else el),
                filter(lambda x: x["type"] in gens, all_gens),
            )
        )
        # ret = list(filter(lambda x: x["type"] in gens, all_gens))
        return ret

    def get_models_car(self, mark):
        models = Db._safe_get(self.car_info, [mark]).keys()
        all_models = self.get_model()
        ret = list(filter(lambda x: x["type"] in models, all_models))
        return ret


if __name__ == "__main__":
    db = Db("training/labels.json", "training/d.json")
    db.get_models_car("BMW")
    print()
