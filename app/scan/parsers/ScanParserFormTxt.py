from CONFIG import DEFAULT_POINTS_COLOR
from app.scan.ScanPoint import ScanPoint
from app.scan.parsers.ScanParserABC import ScanParserABC


class ScanParserFormTxt(ScanParserABC):
    def parse(self, scan):
        with open(self.file_path, "rt", encoding="UTF-8") as file:
            for line in file:
                line = line.strip().split()
                xyz, rgb = self._parse_line(line)
                point = ScanPoint(x=xyz[0], y=xyz[1], z=xyz[2], color=rgb)
                scan.add_point(point)

    @staticmethod
    def _parse_line(line):
        if len(line) in (6, 7):
            xyz = [float(xyz_) for xyz_ in line[:3]]
            rgb = list(map(int, line[3:6]))
        else:
            xyz = [float(xyz_) for xyz_ in line[:3]]
            rgb = DEFAULT_POINTS_COLOR
        return xyz, rgb
