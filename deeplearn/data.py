"""Define dataset"""

import dataclasses
import math
import typing

import numpy
import numpy.random as random
import numpy.typing as ntyping
import torch
import torch.utils.data as data

Float3DArray = numpy.ndarray[tuple[int, int, int], numpy.dtype[numpy.float64]]


@dataclasses.dataclass
class Grid:
    x: Float3DArray
    y: Float3DArray
    z: Float3DArray


@dataclasses.dataclass
class Sphere:
    x: float
    y: float
    z: float
    r: float

    def overlaps(self, other: typing.Self) -> bool:
        distance = math.sqrt(
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2
        )
        return distance < (self.r + other.r)

    def overlaps_any(self, spheres: list[typing.Self]) -> bool:
        for s in spheres:
            if self.overlaps(s):
                return True
        return False

    def to_tensor(self) -> torch.Tensor:
        return torch.Tensor([self.x, self.y, self.z, self.r])


GRID = Grid(*numpy.mgrid[-2:2:40j, -2:2:40j, -2:2:40j])


def create_volumetric_sphere(sphere: Sphere) -> torch.Tensor:
    dist_to_center = numpy.sqrt(
        (GRID.x - sphere.x) ** 2 + (GRID.y - sphere.y) ** 2 + (GRID.z - sphere.z) ** 2
    )
    noise = 0.1 * (random.rand(*GRID.x.shape) - 0.5)
    return torch.from_numpy(  # pyright: ignore[reportUnknownMemberType]
        numpy.exp(-100 * (dist_to_center - sphere.r + noise) ** 2),
    ).to(torch.float)


def create_random_sphere(
    possible_radii: ntyping.NDArray[numpy.float64],
    possible_coordinates: ntyping.NDArray[numpy.float64],
) -> Sphere:
    coordinates: tuple[float, float, float] = tuple(
        random.choice(possible_coordinates, 3)
    )  # pyright: ignore[reportGeneralTypeIssues]
    radius: float = random.choice(possible_radii)
    return Sphere(*coordinates, float(radius))


class SphereDataset(data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, length: int, overlap_allowed: bool = True) -> None:
        super().__init__()
        radii = numpy.linspace(0.2, 0.5, 5)
        coordinate = numpy.linspace(-1, 1, 5)
        self._spheres: list[Sphere] = []
        tries = 0
        while len(self._spheres) < length:
            sphere = create_random_sphere(radii, coordinate)
            if not overlap_allowed and sphere.overlaps_any(self._spheres):
                tries += 1
                continue
            if tries > 100:
                raise ValueError("Could not create a suitable sphere after 20 tries")
            self._spheres.append(sphere)
            tries = 0

    def __len__(self):
        return len(self._spheres)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sphere = self._spheres[index]
        sphere_volume = create_volumetric_sphere(sphere)
        return sphere_volume, sphere.to_tensor()


Loader = data.DataLoader

T = typing.TypeVar("T")


def iter_loader(loader: Loader[T]) -> typing.Iterable[T]:
    """Iterate through a data loader while preserving types."""
    for b in loader:
        batch: T = b
        yield batch
