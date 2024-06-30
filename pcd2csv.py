
'''PCD to CSV Conversion Script for Wheat Rust Disease

This script converts PCD (Point Cloud Data) files related to wheat rust disease to CSV format.
The script processes the binary data in PCD files, extracts the relevant fields, and writes them to a CSV file.

Author: Orly Enrique Apolo-Apolo
e-mail: enrique.apolo@kuleuven.be
Division Forest, Nature and Landscape
Department of Earth & Environmental Sciences
KU Leuven


'''


from __future__ import annotations
import struct
import numpy as np
import pandas as pd
import sys
import re
import string
import random
import csv
import os
from enum import Enum
from io import BufferedReader
from pathlib import Path
from typing import List, Tuple, Union, Literal, Optional
from pydantic import BaseModel, NonNegativeInt, PositiveInt
from tqdm import tqdm

# Type mappings and sizes
PCD_TYPE_TO_STRUCT_FORMAT = {
    ('F', 4): 'f',
    ('F', 8): 'd',
    ('U', 1): 'B',
    ('U', 2): 'H',
    ('U', 4): 'I',
    ('U', 8): 'Q',
    ('I', 1): 'b',
    ('I', 2): 'h',
    ('I', 4): 'i',
    ('I', 8): 'q',
}

PCD_TYPE_TO_NUMPY_TYPE = {
    ('F', 4): np.float32,
    ('F', 8): np.float64,
    ('U', 1): np.uint8,
    ('U', 2): np.uint16,
    ('U', 4): np.uint32,
    ('U', 8): np.uint64,
    ('I', 1): np.int8,
    ('I', 2): np.int16,
    ('I', 4): np.int32,
    ('I', 8): np.int64,
}

HEADER_PATTERN = re.compile(r"(\w+)\s+([\w\s\.\-?\d+\.?\d*]+)")

class Encoding(str, Enum):
    ASCII = "ascii"
    BINARY = "binary"

class MetaData(BaseModel):
    fields: Tuple[str, ...]
    size: Tuple[PositiveInt, ...]
    type: Tuple[Literal["F", "U", "I"], ...]
    count: Tuple[PositiveInt, ...]
    points: NonNegativeInt
    width: NonNegativeInt
    height: NonNegativeInt = 1
    version: str = "0.7"
    viewpoint: Tuple[float, float, float, float, float, float, float] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    data: Encoding = Encoding.BINARY

    @staticmethod
    def parse_header(lines: List[str]) -> MetaData:
        _header = {}
        for line in lines:
            if line.startswith("#") or len(line) < 2: 
                continue

            if (match := re.match(HEADER_PATTERN, line)) is None:
                continue

            value = match.group(2).split()
            key = match.group(1).lower()

            if key in ["version", "data"]:
                _header[key] = value[0]
            elif key in ["width", "height", "points"]:
                _header[key] = int(value[0])
            elif key == "fields":
                _header[key] = tuple(
                    [
                        "#$%&~~" + "".join(random.choices(string.ascii_letters + string.digits, k=6))
                        if s == "_"
                        else s
                        for s in value
                    ]
                )
            elif key in ["type"]:
                _header[key] = tuple(value)
            elif key in ["size", "count"]:
                _header[key] = tuple(int(v) for v in value)
            elif key == "viewpoint":
                _header[key] = tuple(float(v) for v in value)
        
        return MetaData(**_header)

    def build_dtype(self) -> np.dtype:
        field_names: List[str] = []
        np_types: List[np.dtype] = []

        for i, field in enumerate(self.fields):
            np_type = np.dtype(PCD_TYPE_TO_NUMPY_TYPE[(self.type[i], self.size[i])])

            if (count := self.count[i]) == 1:
                field_names.append(field)
                np_types.append(np_type)
            else:
                field_names.extend([f"{field}__{i:04d}" for i in range(count)])
                np_types.extend([np_type] * count)

        return np.dtype([x for x in zip(field_names, np_types)])

class PointCloud:
    def __init__(self, metadata: MetaData, pc_data: np.ndarray) -> None:
        self.metadata = metadata
        self.pc_data = pc_data

    @staticmethod
    def from_fileobj(fp: BufferedReader) -> PointCloud:
        lines: List[str] = []
        for bline in fp:
            if (line := bline.decode(encoding="utf-8").strip()).startswith("#") or not line:
                continue

            lines.append(line)

            if line.startswith("DATA") or len(lines) >= 10:
                break

        metadata = MetaData.parse_header(lines)
        pc_data = _parse_pc_data(fp, metadata)

        fp.seek(0)

        return PointCloud(metadata, pc_data)

    @staticmethod
    def from_path(path: Union[str, Path]) -> PointCloud:
        with open(path, mode="rb") as fp:
            return PointCloud.from_fileobj(fp)

    def numpy(self, fields: Optional[List[str]] = None) -> np.ndarray:
        if fields is None:
            fields = self.fields

        if len(fields) == 0:
            return np.empty((0, 0))

        if self.metadata.points == 0:
            return np.empty((0, len(fields)))

        _stack = tuple(self.pc_data[field] for field in fields)

        return np.vstack(_stack).T

    @property
    def fields(self) -> Tuple[str, ...]:
        fields = []
        for field, count in zip(self.metadata.fields, self.metadata.count):
            if count == 1:
                fields.append(field)
            else:
                fields.extend(f"{field}__{c:04d}" for c in range(count))

        return tuple(fields)

    def __str__(self) -> str:
        return f"PointCloud({self.metadata})"

    def __len__(self) -> int:
        return self.metadata.points

def _parse_pc_data(fp: BufferedReader, metadata: MetaData) -> np.ndarray:
    dtype = metadata.build_dtype()

    if metadata.points > 0:
        if metadata.data == Encoding.ASCII:
            pc_data = np.loadtxt(fp, dtype, delimiter=" ")
        elif metadata.data == Encoding.BINARY:
            buffer = fp.read(metadata.points * dtype.itemsize)
            pc_data = np.frombuffer(buffer, dtype)
    else:
        pc_data = np.empty((0, len(metadata.fields)), dtype)

    return pc_data

# Utility function to convert PCD to CSV
def pcd_to_csv(pcd_path: str, csv_path: str) -> None:
    point_cloud = PointCloud.from_path(pcd_path)
    fields = point_cloud.fields
    points = point_cloud.numpy(fields)

    # Select only the columns x, y, z, and intensity
    selected_fields = ["x", "y", "z", "intensity"]
    selected_indices = [fields.index(field) for field in selected_fields if field in fields]
    selected_points = points[:, selected_indices]

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(selected_fields)
        for row in tqdm(selected_points, desc=f"Processing {pcd_path}", unit="point", colour='green'):
            writer.writerow(row)

# Main function to handle command line arguments and perform conversion
def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py input_directory output_directory")
        sys.exit(-1)
    
    input_directory = Path(sys.argv[1])
    output_directory = Path(sys.argv[2])

    # Ensure output directory exists
    output_directory.mkdir(parents=True, exist_ok=True)

    # Process each PCD file in the input directory
    for input_file in input_directory.glob("*.pcd"):
        try:
            output_file = output_directory / (input_file.stem + '.csv')
            pcd_to_csv(input_file, output_file)
            print(f"Converted {input_file} to {output_file}")
        except Exception as e:
            print(f"Failed to convert {input_file}: {e}")

if __name__ == "__main__":
    main()
