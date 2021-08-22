from pathlib import Path

import numpy as np

from htm_rl.common.plot_utils import plot_grid_images


class BaseOutput:
    save_dir: Path

    def __init__(self, config: dict):
        self.save_dir = config['results_dir']


class ImageOutput(BaseOutput):
    images: list[np.ndarray]
    titles: list[str]
    name_str: str

    def __init__(self, config: dict):
        super().__init__(config)

        self.images = []
        self.titles = []
        self.with_value_text_flags = []

    @property
    def is_empty(self):
        return not self.images

    def handle_img(self, image: np.ndarray, title: str = '', with_value_text: bool = False):
        self.images.append(image.copy())
        self.titles.append(title)
        self.with_value_text_flags.append(with_value_text)

    def flush(self, filename: str):
        if not self.images:
            return

        save_path = self.save_dir.joinpath(filename)
        if not save_path.suffix:
            save_path = save_path.with_suffix('.png')

        plot_grid_images(
            images=self.images, titles=self.titles,
            show=False, save_path=save_path,
            with_value_text_flags=self.with_value_text_flags
        )
        self.images.clear()
        self.titles.clear()
        self.with_value_text_flags.clear()
