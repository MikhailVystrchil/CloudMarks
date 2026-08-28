import laspy
from CONFIG import POINTS_CHUNK_COUNT
from app.scan.ScanPoint import ScanPoint
from app.scan.parsers.ScanParserABC import ScanParserABC


class ScanParserFromLas(ScanParserABC):

    def __init__(self, file_path, chunk_count=POINTS_CHUNK_COUNT):
        super().__init__(file_path)
        self.chunk_count = chunk_count

    @staticmethod
    def __get_xyz(not_scaled_xyz, scales, offsets):
        xyz = []
        for idx, coord in enumerate(not_scaled_xyz):
            xyz.append(float(coord * scales[idx] + offsets[idx]))
        return xyz

    @staticmethod
    def __get_rgb(not_scaled_rgb):
        if not_scaled_rgb is None:
            return [0, 0, 0]
        COLOR_SCALE = 0.003906309605554284  # 1/256
        return [int(r_g_b * COLOR_SCALE) for r_g_b in not_scaled_rgb]

    def parse(self, scan):
        with laspy.open(self.file_path) as input_las:
            for points in input_las.chunk_iterator(self.chunk_count):
                offsets = points.offsets
                scales = points.scales
                points_array = points.array

                has_red = 'red' in points_array.dtype.names
                has_green = 'green' in points_array.dtype.names
                has_blue = 'blue' in points_array.dtype.names

                for point in points_array:
                    xyz = self.__get_xyz((point['X'], point['Y'], point['Z']), offsets=offsets, scales=scales)

                    if has_red and has_green and has_blue:
                        rgb = self.__get_rgb((point['red'], point['green'], point['blue']))
                    else:
                        rgb = [0, 0, 0]

                    point = ScanPoint(x=xyz[0], y=xyz[1], z=xyz[2], color=rgb)
                    scan.add_point(point)
