import pygame
import numpy as np
import math
from typing import List, Dict, Any
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from config import *


class Renderer:
    def __init__(self, render_mode: bool = False, save_video: bool = False):
        self.render_mode = render_mode
        self.save_video = save_video
        self.frame_count = 0
        self.output_dir = "video_frames"
        self.screen = None
        self.clock = None
        self.assets = {}
        self.background_surface = None
        self.debug_view_enabled = False
        if self.render_mode:
            pygame.init()
            self.screen_size = 640
            self.scale = self.screen_size / ARENA_SIZE_M
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            self.clock = pygame.time.Clock()
            self.debug_font = pygame.font.Font(None, 20)
            pygame.display.set_caption("Warehouse AMR Simulation")
            self._load_assets()
            self._generate_background()

        if self.save_video and self.render_mode:
            import shutil

            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)
            os.makedirs(self.output_dir)

    def _load_assets(self):
        try:
            self.assets["robot"] = self._load_image("robot.png", alpha=True)
            # self.assets["person"] = self._load_image("obs_person_1.png", alpha=True)
            # self.assets["obs_robot"] = self._load_image("obs_robot.png", alpha=True)
            self.assets["tile"] = self._load_image("tile.png")
            # self.assets["block"] = self._load_image("block_square.png", alpha=True)
            print("✅ Assets loaded successfully!")
        except (pygame.error, FileNotFoundError) as e:
            print(f"❌ Error loading assets: {e}. Rendering will be disabled.")
            self.assets = {}

    def _load_image(self, filename: str, alpha: bool = False) -> pygame.Surface:
        path = ASSETS_DIR / filename
        image = pygame.image.load(str(path))
        return image.convert_alpha() if alpha else image.convert()

    def render(
        self,
        env_state: Dict[str, Any],
        path: List[np.ndarray] = None,
        history: List[np.ndarray] = None,
    ):
        if not self.render_mode or not self.assets:
            return

        self._draw_background()
        self._draw_static_obstacles(env_state["static_obstacles"])
        self._draw_zones(env_state["start_position"], env_state["goal_position"])

        if path:
            self._draw_path(path)

        if history:
            self._draw_history(history)

        self._draw_obstacles(env_state["obstacle_bodies"])
        self._draw_robot(
            env_state["true_robot_state"],
            env_state["odom_robot_state"],
        )

        if self.debug_view_enabled:
            self._draw_physics_debug(env_state)
            self._draw_lidar_debug(env_state)
            self._draw_dwa_debug(env_state)
            if "lidar_points" in env_state:
                for p in env_state["lidar_points"]:
                    screen_p = self._world_to_screen(p)
                    pygame.draw.circle(self.screen, (0, 255, 255), screen_p, 3)

        # self._draw_ui(env_state["info"])
        self._draw_weight_bars(env_state["info"])
        pygame.display.flip()
        if self.save_video:
            filename = os.path.join(
                self.output_dir, f"frame_{self.frame_count:05d}.png"
            )
            pygame.image.save(self.screen, filename)
            self.frame_count += 1

        self.clock.tick(60)

    def _draw_lidar_debug(self, env_state: Dict[str, Any]):
        robot_state = env_state["true_robot_state"]
        lidar_scan = env_state.get("lidar_scan")

        if lidar_scan is None:
            return

        robot_pos_world = robot_state["position"]
        robot_angle_world = robot_state["angle"]
        robot_pos_screen = self._world_to_screen(robot_pos_world)
        angle_increment = 2 * math.pi / LIDAR_RAYS

        for i, reading in enumerate(lidar_scan):
            ray_angle = robot_angle_world + i * angle_increment
            distance = reading
            endpoint_world = robot_pos_world + np.array(
                [distance * math.cos(ray_angle), distance * math.sin(ray_angle)]
            )
            endpoint_screen = self._world_to_screen(endpoint_world)

            if i == 0:
                color = (255, 50, 50, 150)
                text_color = (255, 100, 100)
            else:
                color = (0, 255, 255, 100)
                text_color = (200, 255, 255)

            pygame.draw.line(self.screen, color, robot_pos_screen, endpoint_screen, 1)
            text_surf = self.debug_font.render(str(i), True, text_color)
            self.screen.blit(text_surf, (endpoint_screen[0] + 5, endpoint_screen[1]))

    def _draw_background(self):
        self.screen.blit(self.background_surface, (0, 0))

    def _draw_history(self, history: List[np.ndarray]):
        if len(history) < 2:
            return
        points = [self._world_to_screen(p) for p in history]
        pygame.draw.lines(self.screen, (0, 100, 255), False, points, 2)

    def _draw_path(self, path: List[np.ndarray]):
        if len(path) < 2:
            return

        path_pixels = [self._world_to_screen(p) for p in path]
        pygame.draw.lines(self.screen, (255, 100, 0, 150), False, path_pixels, 2)
        for p in path_pixels:
            pygame.draw.circle(self.screen, (255, 100, 0), p, 4)

    def _draw_static_obstacles(self, static_obstacles: List[Any]):
        for obs in static_obstacles:
            for fixture in obs.fixtures:
                shape = fixture.shape
                vertices = [(obs.transform * v) * self.scale for v in shape.vertices]
                vertices = [(v[0], self.screen_size - v[1]) for v in vertices]
                pygame.draw.polygon(self.screen, (80, 80, 80), vertices)

    def _draw_zones(self, start_pos: np.ndarray, goal_pos: np.ndarray):
        surface = pygame.Surface((self.screen_size, self.screen_size), pygame.SRCALPHA)
        start_px = self._world_to_screen(start_pos)
        goal_px = self._world_to_screen(goal_pos)

        pygame.draw.circle(
            surface, (100, 100, 255, 85), start_px, int(0.4 * self.scale)
        )
        pygame.draw.circle(surface, (50, 255, 50), goal_px, int(0.4 * self.scale))
        self.screen.blit(surface, (0, 0))

    # def _draw_obstacles(self, obstacle_bodies: List[Any]):
    #     person_img = self.assets["person"]
    #     size = int(OBSTACLE_RADIUS_M * 2 * self.scale)
    #     scaled_img = pygame.transform.scale(person_img, (size, size))

    #     for obstacle in obstacle_bodies:
    #         screen_pos = self._world_to_screen(obstacle.position)
    #         self.screen.blit(scaled_img, scaled_img.get_rect(center=screen_pos))

    def _draw_obstacles(self, obstacle_bodies: List[Any]):
        radius_px = int(OBSTACLE_RADIUS_M * self.scale)
        for obstacle in obstacle_bodies:
            screen_pos = self._world_to_screen(obstacle.position)
            pygame.draw.circle(self.screen, (255, 0, 0), screen_pos, radius_px)
            pygame.draw.circle(self.screen, (0, 0, 0), screen_pos, radius_px, 1)

    def _draw_robot(self, true_state: Dict, odom_state: Dict):
        robot_img = self.assets["robot"]
        size = int(ROBOT_SIZE_M * self.scale)
        scaled_img = pygame.transform.scale(robot_img, (size, size))

        if self.debug_view_enabled:
            ghost_img = scaled_img.copy()
            ghost_img.set_alpha(100)
            rotated_ghost = pygame.transform.rotate(
                ghost_img, np.rad2deg(odom_state["angle"])
            )
            screen_pos_odom = self._world_to_screen(odom_state["position"])
            self.screen.blit(
                rotated_ghost, rotated_ghost.get_rect(center=screen_pos_odom)
            )
            pygame.draw.circle(self.screen, (255, 0, 0, 100), screen_pos_odom, 5, 1)

        rotated_img = pygame.transform.rotate(
            scaled_img, np.rad2deg(true_state["angle"])
        )
        screen_pos_true = self._world_to_screen(true_state["position"])
        self.screen.blit(rotated_img, rotated_img.get_rect(center=screen_pos_true))
        # pygame.draw.circle(self.screen, (0, 255, 0, 100), screen_pos_true, 5, 1)

    def _draw_dwa_debug(self, env_state: Dict[str, Any]):
        endpoints = env_state["info"].get("predicted_endpoints")
        best_endpoint = env_state["info"].get("best_predicted_endpoint")
        robot_pos = env_state["true_robot_state"]["position"]

        if endpoints is None or endpoints.size == 0:
            return

        robot_pos_screen = self._world_to_screen(robot_pos)
        surface = pygame.Surface((self.screen_size, self.screen_size), pygame.SRCALPHA)
        endpoint_list = endpoints.reshape(-1, 2)

        for point in endpoint_list:
            endpoint_screen = self._world_to_screen(point)
            pygame.draw.line(
                surface, (255, 255, 0, 90), robot_pos_screen, endpoint_screen, 1
            )

        self.screen.blit(surface, (0, 0))

        if best_endpoint is not None:
            best_endpoint_screen = self._world_to_screen(best_endpoint)
            pygame.draw.line(
                self.screen, (0, 0, 255, 200), robot_pos_screen, best_endpoint_screen, 3
            )

    def _draw_physics_debug(self, env_state: Dict[str, Any]):
        self._draw_robot_hitbox(env_state["true_robot_state"])
        self._draw_obstacle_hitboxes(env_state["obstacle_bodies"])
        self._draw_static_obs_hitboxes(env_state["static_obstacles"])

    def _draw_robot_hitbox(self, robot_state: Dict):
        center_px = self._world_to_screen(robot_state["position"])
        radius_px = int((ROBOT_SIZE_M / 2) * self.scale)
        pygame.draw.circle(
            self.screen, (255, 0, 0, 150), center_px, radius_px, 2
        )  # Lingkaran merah

    def _draw_obstacle_hitboxes(self, obstacle_bodies: List[Any]):
        for obstacle in obstacle_bodies:
            center_px = self._world_to_screen(obstacle.position)
            radius_px = int(OBSTACLE_RADIUS_M * self.scale)
            pygame.draw.circle(
                self.screen, (255, 255, 0, 150), center_px, radius_px, 2
            )  # Lingkaran kuning

    def _draw_static_obs_hitboxes(self, static_obstacles: List[Any]):
        for obs in static_obstacles:
            for fixture in obs.fixtures:
                shape = fixture.shape
                vertices = [(obs.transform * v) for v in shape.vertices]
                screen_vertices = [self._world_to_screen(v) for v in vertices]
                pygame.draw.polygon(
                    self.screen, (0, 255, 0, 150), screen_vertices, 2
                )  # Poligon hijau

    # def _draw_ui(self, info: Dict):
    #     font = pygame.font.Font(None, 24)
    #     texts = [
    #         # f"Step: {info.get('step_count', 0)}",
    #         # f"True Pos: {info.get('position', np.array([0,0])).round(2)}",
    #         # f"Odom Drift: {info.get('odometry_drift', 0):.3f} m",
    #         # f"Collisions: {info.get('collision_count', 0)}",
    #         # f"Min Obs Dist: {info.get('safety_margin', 0):.2f} m",
    #         # f"Obs Density: {info.get('obs_density', 0):.2f}",
    #         # f"True Vel: {info.get('linear_velocity', 0):.2f} m/s",
    #         # f"Ang Vel: {info.get('angular_velocity', 0):.2f} rad/s",
    #         f"Heading: {info.get('heading_w', 0):.2f}",
    #         f"Velocity: {info.get('velocity_w', 0):.2f}",
    #         f"Clearance: {info.get('clearance_w', 0):.2f}",
    #     ]
    #     for i, text in enumerate(texts):
    #         shadow = font.render(text, True, (0, 0, 0))
    #         self.screen.blit(shadow, (11, 11 + i * 25))
    #         surface = font.render(text, True, (255, 255, 255))
    #         self.screen.blit(surface, (10, 10 + i * 25))

    def _draw_ui(self, info: Dict):
        font = pygame.font.Font(None, 24)
        texts = [
            f"Time: {info.get('distance_traveled', 0) / 1.0:.1f} s",
            f"Vel: {info.get('linear_velocity', 0):.2f} m/s",
            f"Obs Dist: {info.get('safety_margin', 0):.2f} m",
        ]

        bg_rect = pygame.Rect(5, 5, 180, len(texts) * 25 + 10)
        s = pygame.Surface((bg_rect.width, bg_rect.height))
        s.set_alpha(150)
        s.fill((0, 0, 0))
        self.screen.blit(s, (bg_rect.x, bg_rect.y))

        for i, text in enumerate(texts):
            surface = font.render(text, True, (255, 255, 255))
            self.screen.blit(surface, (15, 10 + i * 25))

        self._draw_weight_bars(info)

    def _draw_weight_bars(self, info: Dict):
        w_head = info.get("heading_w", 0.0)
        w_vel = info.get("velocity_w", 0.0)
        w_clear = info.get("clearance_w", 0.0)

        bar_x = self.screen_size - 220
        bar_y = 20  # Posisi Y awal
        bar_w = 150  # Lebar maksimal bar (100%)
        bar_h = 20  # Tinggi per bar
        gap = 25  # Jarak antar bar
        font = pygame.font.Font(None, 20)

        # (Label, Nilai, Warna Bar)
        bars = [
            ("Heading", w_head, (255, 100, 100)),  # Merah
            ("Velocity", w_vel, (100, 100, 255)),  # Biru
            ("Clearance", w_clear, (50, 255, 50)),  # Hijau Terang
        ]

        bg_height = len(bars) * gap + 15
        s = pygame.Surface((210, bg_height))
        s.set_alpha(180)
        s.fill((30, 30, 30))
        self.screen.blit(s, (bar_x - 10, bar_y - 10))

        # title = font.render("FUZZY ADAPTIVE WEIGHTS", True, (255, 255, 255))
        # self.screen.blit(title, (bar_x, bar_y - 25))

        for i, (label, value, color) in enumerate(bars):
            current_y = bar_y + i * gap
            text_surf = font.render(f"{label}", True, (220, 220, 220))
            self.screen.blit(text_surf, (bar_x, current_y - 2))
            pygame.draw.rect(
                self.screen,
                (60, 60, 60),
                (bar_x + 90, current_y, bar_w - 90, bar_h - 5),
            )
            fill_width = int((value / 0.8) * (bar_w - 90))
            fill_width = min(fill_width, bar_w - 90)

            pygame.draw.rect(
                self.screen, color, (bar_x + 90, current_y, fill_width, bar_h - 5)
            )

            val_text = font.render(f"{value:.2f}", True, (255, 255, 255))
            self.screen.blit(val_text, (bar_x + bar_w + 5, current_y - 2))

    def _world_to_screen(self, pos: np.ndarray) -> tuple[int, int]:
        px = int(pos[0] * self.scale)
        py = int((ARENA_SIZE_M - pos[1]) * self.scale)
        return px, py

    def _generate_background(self):
        tile_img = self.assets["tile"]
        self.background_surface = pygame.Surface((self.screen_size, self.screen_size))
        tile_size = tile_img.get_width()
        for x in range(0, self.screen_size, tile_size):
            for y in range(0, self.screen_size, tile_size):
                self.background_surface.blit(tile_img, (x, y))

    def close(self):
        if self.screen is not None:
            pygame.quit()
