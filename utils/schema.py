DATA_SCHEMA = {
    "season": "string",
    "datetime": "string",  # Format: YYYY:MM:DD HH:MM:SS
    "batch_id": "string",
    "image_id": "string",
    "cutout_id": "string",
    "cutout_num": "integer",
    "cutout_height": "integer",
    "cutout_width": "integer",
    "environmental_info": {},
    "cutout_props": {
        "area": "float",
        "eccentricity": "float",
        "solidity": "float",
        "perimeter": "float",
        "green_sum": "integer",
        "blur_effect": "float",
        "num_components": "integer",
        "cropout_rgb_mean": ["float", "float", "float"],
        "cropout_rgb_std": ["float", "float", "float"],
        "is_primary": "boolean",
        "extends_border": "boolean",
    },
}
