import open3d as o3d


class PointCloudWindow:
    """An open3d-based window to display point clouds"""
    def __init__(self):
        """
            Class initialization
        :param width: a positive int: width of the window
        :param height: a positive int: height of the window
        """
        self.pcds = list()

    def load_cloud(self, pcd):
        """
            Load a point cloud
            :param pcd: an open3d.Geometry() object
            :return: None
        """
        self.pcds.append(pcd)

    def show_window(self, width=500, height=500):
        """
            Show the visualization window
            :return: None
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=width, height=height)
        vis.get_render_option().load_from_json("render_option.json")
        for pcd in self.pcds:
            vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()
        self.pcds = list()
