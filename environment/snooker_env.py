"""
Snooker Environment - Physics-based snooker simulation for RL
"""

import numpy as np
import pygame
import pymunk
from pymunk.pygame_util import draw
from gymnasium import spaces
import gymnasium as gym


class Ball:
    """Ball class for snooker balls"""
    def __init__(self, x, y, radius, color, ball_type='red'):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.ball_type = ball_type
        self.pocketed = False
        self.velocity = np.array([0.0, 0.0])
        
    def update(self, body):
        """Update ball position from physics body"""
        self.x = body.position.x
        self.y = body.position.y
        self.velocity = np.array([body.velocity.x, body.velocity.y])
        
    def is_moving(self, threshold=0.5):
        """Check if ball is still moving"""
        speed = np.linalg.norm(self.velocity)
        return speed > threshold


class SnookerEnv(gym.Env):
    """
    Snooker Environment for Reinforcement Learning
    
    Action Space: 
        - angle: [0, 2*pi] - cue angle
        - power: [0, 1] - shot power
        - target_ball: discrete - which ball to target (optional)
    
    Observation Space:
        - Ball positions, velocities, pocket states
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}
    
    def __init__(self, table_width=1200, table_height=600, render_mode=None):
        super().__init__()
        
        self.table_width = table_width
        self.table_height = table_height
        self.render_mode = render_mode
        
        # Physics parameters
        self.friction = 0.02
        self.restitution = 0.9
        self.ball_radius = 12
        self.pocket_radius = 22
        
        # Cushion dimensions
        self.cushion_width = 25
        
        # Pocket positions (6 pockets - 4 corners + 2 middle)
        self.pockets = [
            (self.cushion_width + 5, self.cushion_width + 5),  # Top-left
            (self.table_width // 2, self.cushion_width - 5),   # Top-middle
            (self.table_width - self.cushion_width - 5, self.cushion_width + 5),  # Top-right
            (self.cushion_width + 5, self.table_height - self.cushion_width - 5),  # Bottom-left
            (self.table_width // 2, self.table_height - self.cushion_width + 5),   # Bottom-middle
            (self.table_width - self.cushion_width - 5, self.table_height - self.cushion_width - 5),  # Bottom-right
        ]
        
        # Action space: [angle (continuous), power (continuous)]
        self.action_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([2 * np.pi, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # State dimensions
        self._define_observation_space()
        
        # Initialize
        self.space = None
        self.balls = {}
        self.current_break = 0
        self.total_score = 0
        self.foul_count = 0
        self.max_shots_without_pocket = 3
        
        # Pygame
        self.screen = None
        self.clock = None
        self.font = None
        
    def _define_observation_space(self):
        """Define observation space based on ball positions"""
        # White ball: x, y, vx, vy
        # 15 red balls: x, y for each
        # 6 colored balls: x, y for each
        # Current state info: break, fouls, balls pocketed
        n_features = 2 + 4 + 15 * 2 + 6 * 2 + 4
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(n_features,), dtype=np.float32
        )
        
    def _setup_space(self):
        """Setup PyMunk physics space"""
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = 0.99
        
        # Create cushions (4 walls)
        cushion_thickness = self.cushion_width
        
        # Top cushion
        top = pymunk.Segment(self.space.static_body, 
                           (self.cushion_width, self.cushion_width),
                           (self.table_width - self.cushion_width, self.cushion_width), 
                           cushion_thickness/2)
        top.elasticity = 0.8
        top.friction = 0.5
        
        # Bottom cushion
        bottom = pymunk.Segment(self.space.static_body,
                              (self.cushion_width, self.table_height - self.cushion_width),
                              (self.table_width - self.cushion_width, self.table_height - self.cushion_width),
                              cushion_thickness/2)
        bottom.elasticity = 0.8
        bottom.friction = 0.5
        
        # Left cushion
        left = pymunk.Segment(self.space.static_body,
                            (self.cushion_width, self.cushion_width),
                            (self.cushion_width, self.table_height - self.cushion_width),
                            cushion_thickness/2)
        left.elasticity = 0.8
        left.friction = 0.5
        
        # Right cushion
        right = pymunk.Segment(self.space.static_body,
                             (self.table_width - self.cushion_width, self.cushion_width),
                             (self.table_width - self.cushion_width, self.table_height - self.cushion_width),
                             cushion_thickness/2)
        right.elasticity = 0.8
        right.friction = 0.5
        
        self.space.add(top, bottom, left, right)
        
    def _create_ball(self, x, y, ball_type, color):
        """Create a physics ball"""
        ball = Ball(x, y, self.ball_radius, color, ball_type)
        
        body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, self.ball_radius))
        body.position = (x, y)
        
        shape = pymunk.Circle(body, self.ball_radius)
        shape.elasticity = self.restitution
        shape.friction = self.friction
        
        self.space.add(body, shape)
        self.balls[ball_type] = {'ball': ball, 'body': body, 'shape': shape}
        
        return ball, body
        
    def _setup_initial_positions(self):
        """Setup initial ball positions (simplified snooker layout)"""
        positions = {}
        
        # White ball (cue ball) position
        white_x = self.table_width * 0.25
        white_y = self.table_height / 2
        positions['white'] = (white_x, white_y)
        
        # Red balls (triangle formation)
        red_positions = []
        spacing = self.ball_radius * 2.2
        base_x = self.table_width * 0.7
        base_y = self.table_height / 2
        for row in range(5):
            for col in range(row + 1):
                x = base_x + row * spacing * 0.866
                y = base_y + (col - row / 2) * spacing
                if len(red_positions) < 15:
                    red_positions.append((x, y))
        positions['red'] = red_positions
        
        # Colored balls
        colors_y = self.table_height * 0.25
        yellow_x = self.table_width * 0.25
        positions['yellow'] = (yellow_x, colors_y)
        
        green_x = self.table_width * 0.25
        positions['green'] = (green_x, self.table_height - colors_y)
        
        brown_x = self.table_width * 0.25
        positions['brown'] = (brown_x, self.table_height / 2)
        
        blue_x = self.table_width / 2
        positions['blue'] = (blue_x, self.table_height / 2)
        
        pink_x = self.table_width * 0.6
        pink_y = self.table_height / 2
        positions['pink'] = (pink_x, pink_y)
        
        black_x = self.table_width * 0.8
        black_y = self.table_height / 2
        positions['black'] = (black_x, black_y)
        
        return positions
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Clear previous state
        if self.space:
            for ball_type in list(self.balls.keys()):
                ball_data = self.balls[ball_type]
                self.space.remove(ball_data['body'], ball_data['shape'])
        self.balls = {}
        
        # Setup physics space
        self._setup_space()
        
        # Setup ball positions
        positions = self._setup_initial_positions()
        
        # Create white ball
        x, y = positions['white']
        self._create_ball(x, y, 'white', (255, 255, 255))
        
        # Create red balls
        for i, (x, y) in enumerate(positions['red']):
            self._create_ball(x, y, f'red_{i}', (220, 20, 20))
        
        # Create colored balls
        color_map = {
            'yellow': (255, 255, 0),
            'green': (0, 180, 0),
            'brown': (139, 69, 19),
            'blue': (30, 144, 255),
            'pink': (255, 192, 203),
            'black': (30, 30, 30)
        }
        
        for ball_type, pos in positions.items():
            if ball_type not in ['white', 'red'] and isinstance(pos, tuple):
                color = color_map.get(ball_type, (128, 128, 128))
                self._create_ball(pos[0], pos[1], ball_type, color)
        
        # Reset game state
        self.current_break = 0
        self.total_score = 0
        self.foul_count = 0
        self.shots_without_pocket = 0
        self.last_pocketed_balls = []
        
        # Initialize pygame for rendering
        if self.render_mode == 'human':
            self._init_pygame()
            
        return self._get_obs(), {}
        
    def _init_pygame(self):
        """Initialize pygame for rendering"""
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.table_width, self.table_height))
            pygame.display.set_caption("Snooker RL")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
            
    def _get_obs(self):
        """Get current observation"""
        obs = []
        
        # White ball info (x, y, vx, vy)
        white = self.balls.get('white')
        if white:
            ball = white['ball']
            body = white['body']
            obs.extend([
                (ball.x / self.table_width) * 2 - 1,
                (ball.y / self.table_height) * 2 - 1,
                body.velocity.x / 100,
                body.velocity.y / 100
            ])
        else:
            obs.extend([0, 0, 0, 0])
            
        # Red balls info (x, y for each)
        red_count = 0
        for i in range(15):
            red_key = f'red_{i}'
            if red_key in self.balls:
                ball = self.balls[red_key]['ball']
                if not ball.pocketed:
                    obs.extend([
                        (ball.x / self.table_width) * 2 - 1,
                        (ball.y / self.table_height) * 2 - 1
                    ])
                    red_count += 1
                else:
                    obs.extend([-1, -1])
            else:
                obs.extend([-1, -1])
                
        # Colored balls info (x, y for each)
        colored_balls = ['yellow', 'green', 'brown', 'blue', 'pink', 'black']
        for ball_type in colored_balls:
            if ball_type in self.balls:
                ball = self.balls[ball_type]['ball']
                if not ball.pocketed:
                    obs.extend([
                        (ball.x / self.table_width) * 2 - 1,
                        (ball.y / self.table_height) * 2 - 1
                    ])
                else:
                    obs.extend([-1, -1])
            else:
                obs.extend([-1, -1])
                
        # Additional state info
        obs.extend([
            self.current_break / 100,
            self.foul_count / 10,
            self.shots_without_pocket / self.max_shots_without_pocket,
            len([b for b in self.balls.values() if not b['ball'].pocketed]) / 22
        ])
        
        return np.array(obs, dtype=np.float32)
        
    def step(self, action):
        """Execute one step in the environment"""
        angle, power = action
        
        # Apply shot
        self._apply_shot(angle, power)
        
        # Simulate until all balls stop
        self._simulate_until_stable()
        
        # Check for fouls and rewards
        reward, done = self._compute_reward_and_done()
        
        # Update state
        obs = self._get_obs()
        info = {
            'break': self.current_break,
            'score': self.total_score,
            'foul': self.foul_count > 0,
            'pocketed': self.last_pocketed_balls
        }
        
        return obs, reward, done, False, info
        
    def _apply_shot(self, angle, power):
        """Apply shot to white ball"""
        if 'white' not in self.balls:
            return
            
        white_body = self.balls['white']['body']
        white_ball = self.balls['white']['ball']
        
        # Check if white ball is pocketed (foul)
        if white_ball.pocketed:
            # Reset white ball to starting position
            white_body.position = (self.table_width * 0.25, self.table_height / 2)
            white_body.velocity = (0, 0)
            white_ball.pocketed = False
            self.foul_count += 1
            
        # Calculate shot direction
        velocity_magnitude = power * 2000 + 200
        vx = np.cos(angle) * velocity_magnitude
        vy = np.sin(angle) * velocity_magnitude
        
        white_body.velocity = (vx, vy)
        
    def _simulate_until_stable(self, max_steps=500):
        """Simulate until all balls are stable"""
        self.last_pocketed_balls = []
        
        for step in range(max_steps):
            self.space.step(1/60)
            
            # Update ball positions
            for ball_type, ball_data in self.balls.items():
                ball_data['ball'].update(ball_data['body'])
                
            # Check for pocketed balls
            for ball_type, ball_data in self.balls.items():
                if not ball_data['ball'].pocketed:
                    if self._is_ball_pocketed(ball_data['ball']):
                        self._handle_pocketed_ball(ball_type, ball_data['ball'])
                        
            # Check if all balls are stable
            if self._all_balls_stable():
                break
                
    def _is_ball_pocketed(self, ball):
        """Check if a ball is in a pocket"""
        for pocket_x, pocket_y in self.pockets:
            distance = np.sqrt((ball.x - pocket_x)**2 + (ball.y - pocket_y)**2)
            if distance < self.pocket_radius:
                return True
        return False
        
    def _handle_pocketed_ball(self, ball_type, ball):
        """Handle a pocketed ball"""
        ball.pocketed = True
        ball.velocity = np.array([0, 0])
        self.last_pocketed_balls.append(ball_type)
        
        # Move ball off table
        ball_data = self.balls[ball_type]
        ball_data['body'].position = (-100, -100)
        ball_data['body'].velocity = (0, 0)
        
    def _all_balls_stable(self, threshold=0.5):
        """Check if all balls are stable (not moving)"""
        for ball_type, ball_data in self.balls.items():
            if not ball_data['ball'].pocketed:
                if ball_data['ball'].is_moving(threshold):
                    return False
        return True
        
    def _compute_reward_and_done(self):
        """Compute reward and check if episode is done"""
        reward = 0
        done = False
        
        pocketed_this_shot = self.last_pocketed_balls.copy()
        
        # White ball pocketed (foul)
        if 'white' in pocketed_this_shot:
            reward -= 10
            self.foul_count += 1
            
        # Count colored balls
        colored_count = sum(1 for b in pocketed_this_shot 
                          if b in ['yellow', 'green', 'brown', 'blue', 'pink', 'black'])
        
        # Red balls (2 points each)
        red_count = sum(1 for b in pocketed_this_shot if b.startswith('red_'))
        reward += red_count * 2
        self.current_break += red_count * 2
        
        # Colored balls
        color_values = {
            'yellow': 2, 'green': 3, 'brown': 4, 
            'blue': 5, 'pink': 6, 'black': 7
        }
        for ball_type in pocketed_this_shot:
            if ball_type in color_values:
                value = color_values[ball_type]
                reward += value
                self.current_break += value
                
        # Track shots without pocket
        if red_count > 0 or colored_count > 0:
            self.shots_without_pocket = 0
        else:
            self.shots_without_pocket += 1
            
        # Check for foul (white ball pocketed)
        if 'white' in pocketed_this_shot:
            self.current_break = 0
            reward -= 5
            
        # End episode conditions
        remaining_balls = [b for b in self.balls.values() 
                         if not b['ball'].pocketed and not b['ball'].ball_type == 'white']
        
        # Episode ends if no balls remaining or too many fouls
        if len(remaining_balls) == 0:
            # All balls cleared!
            reward += 100
            self.total_score += self.current_break
            done = True
            
        elif self.shots_without_pocket >= self.max_shots_without_pocket:
            # No progress made
            self.current_break = 0
            done = True
            
        elif self.foul_count >= 3:
            # Too many fouls
            done = True
            
        # Small negative reward for each step (encourage efficiency)
        reward -= 0.1
        
        return reward, done
        
    def render(self):
        """Render the environment"""
        if self.render_mode == 'human':
            return self._render_human()
        elif self.render_mode == 'rgb_array':
            return self._render_rgb_array()
            
    def _render_human(self):
        """Render using pygame"""
        if self.screen is None:
            self._init_pygame()
            
        self.screen.fill((34, 139, 34))  # Green felt
            
        # Draw cushions
        cushion_color = (0, 100, 0)
        pygame.draw.rect(self.screen, cushion_color, 
                        (0, 0, self.table_width, self.cushion_width))
        pygame.draw.rect(self.screen, cushion_color,
                        (0, self.table_height - self.cushion_width, self.table_width, self.cushion_width))
        pygame.draw.rect(self.screen, cushion_color,
                        (0, 0, self.cushion_width, self.table_height))
        pygame.draw.rect(self.screen, cushion_color,
                        (self.table_width - self.cushion_width, 0, self.cushion_width, self.table_height))
        
        # Draw pockets
        for pocket_x, pocket_y in self.pockets:
            pygame.draw.circle(self.screen, (0, 0, 0), 
                             (int(pocket_x), int(pocket_y)), 
                             self.pocket_radius)
                             
        # Draw balls
        for ball_type, ball_data in self.balls.items():
            ball = ball_data['ball']
            if not ball.pocketed:
                pygame.draw.circle(self.screen, ball.color,
                                 (int(ball.x), int(ball.y)),
                                 self.ball_radius)
                pygame.draw.circle(self.screen, (255, 255, 255),
                                 (int(ball.x), int(ball.y)),
                                 self.ball_radius, 1)
                                 
        # Draw UI
        text_color = (255, 255, 255)
        break_text = self.font.render(f"Break: {self.current_break}", True, text_color)
        score_text = self.font.render(f"Score: {self.total_score}", True, text_color)
        foul_text = self.font.render(f"Fouls: {self.foul_count}", True, text_color)
        
        self.screen.blit(break_text, (10, 10))
        self.screen.blit(score_text, (10, 50))
        self.screen.blit(foul_text, (10, 90))
        
        pygame.display.flip()
        self.clock.tick(self.metadata['render_fps'])
        
    def _render_rgb_array(self):
        """Return RGB array for rendering"""
        if self.screen is None:
            self._init_pygame()
            
        self._render_human()
        return pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)
        
    def close(self):
        """Clean up resources"""
        if self.screen:
            pygame.quit()
            self.screen = None
            
    def get_available_balls(self):
        """Get list of non-pocketed balls"""
        available = []
        for ball_type, ball_data in self.balls.items():
            if not ball_data['ball'].pocketed:
                available.append(ball_type)
        return available
