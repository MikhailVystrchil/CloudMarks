from app.scan.Scan import Scan


class ScanSplitterByLabels:
    """
    Разделяет скан на подсканы по меткам (например, по DBSCAN-кластерам
    или по классам направлений нормалей).
    """

    def __init__(self, scan, label_attr="labels", include_noise=False, noise_label=-1):
        self.scan = scan
        self.label_attr = label_attr
        self.include_noise = include_noise
        self.noise_label = noise_label

    def split(self):
        labels_set = set()
        for p in self.scan:
            if not hasattr(p, self.label_attr):
                continue
            lbl = getattr(p, self.label_attr)
            if (lbl == self.noise_label) and (not self.include_noise):
                continue
            labels_set.add(lbl)

        label_to_scan = {}
        for lbl in labels_set:
            sub_scan = Scan(scan_name=f"{self.scan.name}_label_{lbl}")
            label_to_scan[lbl] = sub_scan

        for p in self.scan:
            if not hasattr(p, self.label_attr):
                continue
            lbl = getattr(p, self.label_attr)
            if (lbl == self.noise_label) and (not self.include_noise):
                continue
            sub_scan = label_to_scan.get(lbl)
            if sub_scan is not None:
                sub_scan.add_point(p)

        for sub_scan in label_to_scan.values():
            sub_scan.borders = sub_scan._get_borders_dict(sub_scan._points)

        return label_to_scan
