"""Define dataset"""

import dataclasses
import math
import typing

import torch
import torch.utils.data as data

Loader = data.DataLoader

TorchTuple = tuple[torch.Tensor, torch.Tensor]

T = typing.TypeVar("T")

@dataclasses.dataclass
class Grid:
    x: torch.Tensor
    y: torch.Tensor
    z: torch.Tensor

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

class SphereDataset(data.Dataset[TorchTuple]):
    def __init__(self, spheres: torch.Tensor, volumes: torch.Tensor) -> None:
        super().__init__()
        self._spheres = spheres
        self._volumes = volumes

    def __len__(self):
        return len(self._spheres)

    def __getitem__(self, index: int) -> TorchTuple:
        return self._volumes[index], self._spheres[index] 
    
    @classmethod
    def generate(cls, size: int):
        """Generate a dataset with a specific size."""
        spheres = create_n_spheres(n=size)
        volumes = create_volumes(spheres=spheres)
        return cls(spheres=spheres, volumes=volumes)
    
    @classmethod
    def from_file(cls, filename: str):
        spheres = torch.load(  # pyright: ignore[reportUnknownMemberType]
            filename + ".X.data"
        )
        volumes = torch.load(  # pyright: ignore[reportUnknownMemberType]
            filename + ".Y.data"
        )
        return cls(spheres=spheres, volumes=volumes)
    
    def to_file(self, filename: str):
        torch.save(self._spheres, filename + ".X.data")  # pyright: ignore[reportUnknownMemberType]
        torch.save(self._volumes, filename + ".Y.data")  # pyright: ignore[reportUnknownMemberType]

def create_n_spheres(n: int)->torch.Tensor:
    random_values = torch.rand((n, 4))
    diff = torch.tensor([0.5, 0.5, 0.5, 0]).repeat(n,1)
    return random_values - diff


def create_grid(min_value: int, max_value: int, res:int)->Grid:
    grid_axis = torch.linspace(min_value, max_value, res)
    grid_tuple = torch.meshgrid([grid_axis, grid_axis, grid_axis], indexing="xy")
    return Grid(*grid_tuple)

GRID_RESOLUTION = 40
GRID = create_grid(-2, 2, GRID_RESOLUTION)

def create_volumes(spheres: torch.Tensor)->torch.Tensor:
    shape = spheres.shape
    assert len(shape) == 2
    assert shape[1] == 4
    n = shape[0]
    result_shape = (n,GRID_RESOLUTION, GRID_RESOLUTION, GRID_RESOLUTION)
    coordinate_shape = (n,1,1,1)
    X = GRID.x.repeat(*coordinate_shape)
    Y = GRID.y.repeat(*coordinate_shape)
    Z = GRID.z.repeat(*coordinate_shape)
    cX = spheres[:, 0].reshape(coordinate_shape)
    cY = spheres[:, 1].reshape(coordinate_shape)
    cZ = spheres[:, 2].reshape(coordinate_shape)
    cR = spheres[:, 3].reshape(coordinate_shape)
    dist = torch.sqrt((X-cX)**2+(Y-cY)**2+(Z-cZ)**2) - cR
    result = torch.exp(- 100 * dist ** 2)
    assert result.shape == result_shape
    return result

def iter_loader(loader: Loader[T]) -> typing.Iterable[T]:
    """Iterate through a data loader while preserving types."""
    for b in loader:
        batch: T = b
        yield batch
