from .nerf import NeRFDataset
from .nsvf import NSVFDataset
from .colmap import ColmapDataset
from .nerfpp import NeRFPPDataset
from .scannet import ScanNetDataset
from .scannetpp import ScanNetPPDataset
from ._google_scanned_obj import GoogleScannedDataset


dataset_dict = {'nerf': NeRFDataset,
                'nsvf': NSVFDataset,
                'colmap': ColmapDataset,
                'nerfpp': NeRFPPDataset,
                'scannet': ScanNetDataset,
                'scannetpp': ScanNetPPDataset,
                'google_scanned': GoogleScannedDataset}