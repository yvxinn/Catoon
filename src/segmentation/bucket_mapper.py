"""
BucketMapper - 语义类别到桶的映射

根据 ADE20K 的 id2label 自动构建类别桶映射。
不硬编码 ID，按 label 名称进行聚类。
"""

from enum import Enum
from typing import Any


class SemanticBucket(str, Enum):
    """语义桶枚举"""
    SKY = "SKY"
    PERSON = "PERSON"
    VEGETATION = "VEGETATION"
    BUILDING = "BUILDING"
    ROAD = "ROAD"
    WATER = "WATER"
    OTHERS = "OTHERS"


# 每个桶包含的关键词（用于从 label 名称映射）
BUCKET_KEYWORDS: dict[SemanticBucket, list[str]] = {
    SemanticBucket.SKY: ["sky"],
    SemanticBucket.PERSON: ["person", "people"],
    SemanticBucket.VEGETATION: [
        "tree", "grass", "plant", "vegetation", "flower", 
        "palm", "bush", "forest", "field", "leaves", "flora"
    ],
    SemanticBucket.BUILDING: [
        "building", "house", "skyscraper", "wall", "window", "door",
        "tower", "bridge", "fence", "gate", "awning", "column",
        "arcade", "canopy", "roof", "ceiling", "floor"
    ],
    SemanticBucket.ROAD: [
        "road", "sidewalk", "path", "earth", "ground", "floor",
        "pavement", "dirt", "track", "runway", "platform", "stairs",
        "stairway", "escalator"
    ],
    SemanticBucket.WATER: [
        "water", "sea", "river", "lake", "pool", "waterfall",
        "ocean", "pond", "stream", "fountain"
    ],
}


class BucketMapper:
    """类别到桶的映射器"""
    
    def __init__(self, id2label: dict[int, str] | None = None):
        """
        初始化映射器
        
        Args:
            id2label: 从模型配置获取的 ID 到标签名映射
                      如果为 None，则使用默认的 ADE20K 映射
        """
        self.id2label = id2label or self._get_default_ade20k_id2label()
        self.id2bucket = self._build_id2bucket_mapping()
        self.bucket2ids = self._build_bucket2ids_mapping()
    
    def _build_id2bucket_mapping(self) -> dict[int, SemanticBucket]:
        """
        构建 ID 到桶的映射
        
        Returns:
            {class_id: bucket} 映射
        """
        id2bucket = {}
        
        for class_id, label_name in self.id2label.items():
            label_lower = label_name.lower()
            bucket = self._label_to_bucket(label_lower)
            id2bucket[class_id] = bucket
        
        return id2bucket
    
    def _build_bucket2ids_mapping(self) -> dict[SemanticBucket, list[int]]:
        """
        构建桶到 ID 列表的映射
        
        Returns:
            {bucket: [class_ids]} 映射
        """
        bucket2ids: dict[SemanticBucket, list[int]] = {
            bucket: [] for bucket in SemanticBucket
        }
        
        for class_id, bucket in self.id2bucket.items():
            bucket2ids[bucket].append(class_id)
        
        return bucket2ids
    
    def _label_to_bucket(self, label: str) -> SemanticBucket:
        """
        将标签名映射到桶
        
        Args:
            label: 小写的标签名
        
        Returns:
            对应的语义桶
        """
        for bucket, keywords in BUCKET_KEYWORDS.items():
            for keyword in keywords:
                if keyword in label:
                    return bucket
        
        return SemanticBucket.OTHERS
    
    def get_bucket(self, class_id: int) -> SemanticBucket:
        """
        获取类别 ID 对应的桶
        
        Args:
            class_id: 类别 ID
        
        Returns:
            语义桶
        """
        return self.id2bucket.get(class_id, SemanticBucket.OTHERS)
    
    def get_bucket_ids(self, bucket: SemanticBucket | str) -> list[int]:
        """
        获取桶包含的所有类别 ID
        
        Args:
            bucket: 语义桶或桶名称字符串
        
        Returns:
            类别 ID 列表
        """
        if isinstance(bucket, str):
            bucket = SemanticBucket(bucket)
        return self.bucket2ids.get(bucket, [])
    
    def get_all_buckets(self) -> list[SemanticBucket]:
        """获取所有桶"""
        return list(SemanticBucket)
    
    def print_mapping(self) -> None:
        """打印映射关系（调试用）"""
        print("=" * 60)
        print("Bucket Mapping")
        print("=" * 60)
        for bucket in SemanticBucket:
            ids = self.bucket2ids[bucket]
            labels = [self.id2label.get(i, "?") for i in ids[:5]]
            more = f"... (+{len(ids)-5})" if len(ids) > 5 else ""
            print(f"{bucket.value:12} ({len(ids):3} classes): {labels}{more}")
    
    @staticmethod
    def _get_default_ade20k_id2label() -> dict[int, str]:
        """
        获取默认的 ADE20K id2label 映射
        
        注意：这是一个简化版本，完整版本会从模型配置自动获取
        """
        # ADE20K 150 类的部分常用类别
        # 完整映射会在加载 SegFormer 模型时自动获取
        return {
            0: "wall",
            1: "building",
            2: "sky",
            3: "floor",
            4: "tree",
            5: "ceiling",
            6: "road",
            7: "bed",
            8: "windowpane",
            9: "grass",
            10: "cabinet",
            11: "sidewalk",
            12: "person",
            13: "earth",
            14: "door",
            15: "table",
            16: "mountain",
            17: "plant",
            18: "curtain",
            19: "chair",
            20: "car",
            21: "water",
            22: "painting",
            23: "sofa",
            24: "shelf",
            25: "house",
            26: "sea",
            27: "mirror",
            28: "rug",
            29: "field",
            30: "armchair",
            31: "seat",
            32: "fence",
            33: "desk",
            34: "rock",
            35: "wardrobe",
            36: "lamp",
            37: "bathtub",
            38: "railing",
            39: "cushion",
            40: "base",
            41: "box",
            42: "column",
            43: "signboard",
            44: "chest",
            45: "counter",
            46: "sand",
            47: "sink",
            48: "skyscraper",
            49: "fireplace",
            50: "refrigerator",
            51: "grandstand",
            52: "path",
            53: "stairs",
            54: "runway",
            55: "case",
            56: "pool",
            57: "pillow",
            58: "screen",
            59: "stairway",
            60: "river",
            61: "bridge",
            62: "bookcase",
            63: "blind",
            64: "coffee",
            65: "toilet",
            66: "flower",
            67: "book",
            68: "hill",
            69: "bench",
            70: "countertop",
            71: "stove",
            72: "palm",
            73: "kitchen",
            74: "computer",
            75: "swivel",
            76: "boat",
            77: "bar",
            78: "arcade",
            79: "hovel",
            80: "bus",
            81: "towel",
            82: "light",
            83: "truck",
            84: "tower",
            85: "chandelier",
            86: "awning",
            87: "streetlight",
            88: "booth",
            89: "television",
            90: "airplane",
            91: "dirt",
            92: "apparel",
            93: "pole",
            94: "land",
            95: "bannister",
            96: "escalator",
            97: "ottoman",
            98: "bottle",
            99: "buffet",
            100: "poster",
            101: "stage",
            102: "van",
            103: "ship",
            104: "fountain",
            105: "conveyer",
            106: "canopy",
            107: "washer",
            108: "plaything",
            109: "swimming",
            110: "stool",
            111: "barrel",
            112: "basket",
            113: "waterfall",
            114: "tent",
            115: "bag",
            116: "minibike",
            117: "cradle",
            118: "oven",
            119: "ball",
            120: "food",
            121: "step",
            122: "tank",
            123: "trade",
            124: "microwave",
            125: "pot",
            126: "animal",
            127: "bicycle",
            128: "lake",
            129: "dishwasher",
            130: "screen",
            131: "blanket",
            132: "sculpture",
            133: "hood",
            134: "sconce",
            135: "vase",
            136: "traffic",
            137: "tray",
            138: "ashcan",
            139: "fan",
            140: "pier",
            141: "crt",
            142: "plate",
            143: "monitor",
            144: "bulletin",
            145: "shower",
            146: "radiator",
            147: "glass",
            148: "clock",
            149: "flag",
        }

