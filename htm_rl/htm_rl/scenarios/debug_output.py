from pathlib import Path

import numpy as np


class BaseOutput:
    save_dir: Path

    def __init__(self, config: dict):
        self.save_dir = config['results_dir']


class ImageOutput(BaseOutput):
    images: list[np.ndarray]
    titles: list[str]
    with_value_text_flags: list[bool]
    name_str: str

    def __init__(self, config: dict):
        super().__init__(config)
        self.reset()

    def reset(self):
        self.images = []
        self.titles = []
        self.with_value_text_flags = []

    def restore(self, images, titles, with_value_text_flags, **kwargs):
        if self.is_empty:
            self.images = images
            self.titles = titles
            self.with_value_text_flags = with_value_text_flags
        else:
            self.images.extend(images)
            self.titles.extend(titles)
            self.with_value_text_flags.extend(with_value_text_flags)

    @property
    def is_empty(self):
        return not self.images

    def handle_img(self, image: np.ndarray, title: str = '', with_value_text: bool = False):
        self.images.append(image.copy())
        self.titles.append(title)
        self.with_value_text_flags.append(with_value_text)

    def flush(self, filename: str = None, save_path: str = None):
        if not self.images:
            return

        if filename is not None and save_path is None:
            save_path = self.save_dir.joinpath(filename)
            if not save_path.suffix:
                save_path = save_path.with_suffix('.png')

        output = {
            'images': self.images,
            'titles': self.titles,
            'with_value_text_flags': self.with_value_text_flags,
        }
        if save_path is not None:
            output['save_path'] = save_path

        self.reset()
        return output
