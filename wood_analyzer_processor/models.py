from __future__ import annotations
from typing import List, Tuple, Dict

from pydantic import BaseModel
import numpy as np


def encode_rle(array: np.ndarray) -> str:
    last_label = 0
    count = 0
    encoded_string = ""
    for label in np.nditer(array, order="C"):
        label = label.item()
        if label == last_label:
            count += 1
        else:
            encoded_string += str(count) + "," + str(last_label) + "|"
            last_label = label
            count = 1

    encoded_string += str(count) + "," + str(last_label) + "|"
    if encoded_string.startswith("0,0|"):
        encoded_string = encoded_string[4:]

    return encoded_string


def decode_rle(encoded_string: str, shape: Tuple[int, int]) -> np.ndarray:
    def get_token(string: str):
        i = 0
        while i < len(string):
            j = i
            while string[j] != ",":
                j += 1
            count = int(string[i:j])
            j += 1
            i = j
            while string[j] != "|":
                j += 1
            label = int(string[i:j])
            yield count, label
            i = j + 1

    array = np.empty(shape=(shape[0] * shape[1],), dtype=np.uint8)
    i = 0
    for count, label in get_token(encoded_string):
        array[i : i + count] = label
        i += count

    return array.reshape(shape)


def to_camel(string: str):
    string = "".join(word.capitalize() for word in string.split("_"))  # to pascal
    return "".join([string[:1].lower(), string[1:]])  # to camel


class CamelBaseModel(BaseModel):
    class Config:
        alias_generator = to_camel


class AreaPerRing(CamelBaseModel):
    ring_area: int
    duct_area: int


class Areas(CamelBaseModel):
    resin_ducts_area: int
    resin_ducts_count: int
    areas_per_ring: Dict[str, AreaPerRing]


class Segmentation(CamelBaseModel, arbitrary_types_allowed=True):
    segmentation_array: np.ndarray
    rings_array: np.ndarray = None
    width: int
    height: int
    measure_segments: List[List[int]] = []
    areas: Areas = None
    ppi: int = 400

    @staticmethod
    def decode(encoded_object: dict) -> Segmentation:
        encoded_object["segmentationArray"] = decode_rle(
            encoded_object["segmentationArray"],
            shape=(
                encoded_object["height"],
                encoded_object["width"],
            ),
        )
        if encoded_object["ringsArray"] is not None:
            encoded_object["ringsArray"] = decode_rle(
                encoded_object["ringsArray"],
                shape=(
                    encoded_object["height"],
                    encoded_object["width"],
                ),
            )

        return Segmentation.parse_obj(encoded_object)

    @staticmethod
    def from_array(segmentation_array: np.ndarray) -> Segmentation:
        return Segmentation(
            segmentationArray=segmentation_array,
            width=segmentation_array.shape[1],
            height=segmentation_array.shape[0],
        )

    def encode(self) -> dict:
        encoded_object = self.dict(by_alias=True)

        encoded_object["segmentationArray"] = encode_rle(self.segmentation_array)
        if self.rings_array is not None:
            encoded_object["ringsArray"] = encode_rle(self.rings_array)

        return encoded_object

    def add_areas(
        self,
        tree_rings_areas: dict,
        ducts_areas_per_ring: dict,
        resin_ducts_area: int,
        resin_ducts_count: int,
    ):
        areas_per_ring = {}
        for label, ring in tree_rings_areas.items():
            apr = {
                "ringArea": ring,
                "ductArea": ducts_areas_per_ring.get(label, 0),
            }
            areas_per_ring[str(label)] = apr

        obj = {
            "resinDuctsArea": resin_ducts_area,
            "resinDuctsCount": resin_ducts_count,
            "areasPerRing": areas_per_ring,
        }
        self.areas = Areas.parse_obj(obj=obj)
