import pygame
import cv2
import numpy as np
from collections import Counter
import random
import threading

# Initialize Pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 800, 800
GRID_SIZE = 80
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")


# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Set the desired width and height for resizing
desired_width = 320
desired_height = 240

roi_width = 250  # Width of the ROI
roi_height = 180  # Height of the ROI
roi_x = (desired_width - roi_width) // 2  # X coordinate of the top-left corner of the ROI
roi_y = (desired_height - roi_height) // 2  # Y coordinate of the top-left corner of the ROI


# Parameters for Farneback optical flow
fb_params = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=10,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0
)

# Size of patches
patch_size = 10

# Movement threshold
movement_threshold = 5

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Read the first frame
ret, first_frame = cap.read()
if not ret:
    print("Failed to read from camera. Exiting...")
    cap.release()
    exit()

# Resize and convert the first frame to grayscale
first_frame_resized = cv2.resize(first_frame, (desired_width, desired_height))
prev_gray = cv2.cvtColor(first_frame_resized, cv2.COLOR_BGR2GRAY)

predictions = []

def draw_optical_flow(screen, flow, patch_size):
    for y in range(0, flow.shape[0], patch_size):
        for x in range(0, flow.shape[1], patch_size):
            fx, fy = flow[y, x]
            magnitude = np.sqrt(fx ** 2 + fy ** 2)
            if magnitude > movement_threshold:
                end_point = (x + int(fx), y + int(fy))
                cv2.arrowedLine(screen, (x, y), end_point, (0, 255, 0), 1)

def most_common_value(arr):
    # Create a Counter object from the array
    counter = Counter(arr)

    # Find the most common value
    most_common = counter.most_common(1)

    # Return the most common value
    return most_common[0][0] if most_common else None


def get_next():
    global predictions
    global prev_gray
    iterations = 0
    while True:
        iterations += 1
        # Capture the next frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            exit(10)

        # Resize and convert the current frame to grayscale
        frame_resized = cv2.resize(frame, (desired_width, desired_height))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        roi = gray[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width], roi, None,
                                            **fb_params)

        # Variables to accumulate the flow direction
        total_flow_x = 0
        total_flow_y = 0
        count = 0

        # Draw arrows
        for y in range(0, flow.shape[0], patch_size):
            for x in range(0, flow.shape[1], patch_size):
                fx, fy = flow[y, x]
                magnitude = np.sqrt(fx ** 2 + fy ** 2)
                if magnitude > movement_threshold:
                    total_flow_x += fx
                    total_flow_y += fy
                    count += 1

        # Calculate average flow direction
        if count > 0:
            avg_flow_x = total_flow_x / count
            avg_flow_y = total_flow_y / count

            threshold = 4

            direction = None

            print(avg_flow_x, avg_flow_y)

            if abs(avg_flow_x) > abs(avg_flow_y) and abs(avg_flow_x) > threshold:
                if avg_flow_x > 0:
                    direction = "RIGHT"
                else:
                    direction = "LEFT"
            elif abs(avg_flow_y) > threshold:
                if avg_flow_y > 0 and avg_flow_y > 5:
                    direction = "DOWN"
                else:
                    direction = "UP"

            if direction is not None:
                predictions.append(direction)

        if len(predictions) > 10:
            predictions = predictions[-10:]

        prev_gray = gray

# Start the thread for continuous prediction
thread = threading.Thread(target=get_next)
thread.daemon = True
thread.start()

# Snake class
class Snake:
    def __init__(self):
        self.body = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.l = 3
        self.direction = (1, 0)

    def move(self):
        x, y = self.body[0]
        dx, dy = self.direction
        new_x = (x + dx) % GRID_WIDTH
        new_y = (y + dy) % GRID_HEIGHT
        if (new_x, new_y) in self.body[1:]:
            return False  # Game over if snake collides with itself
        self.body.insert(0, (new_x, new_y))
        if len(self.body) > self.l:
            self.body = self.body[:self.l]
        return True

    def change_direction(self, direction):
        if direction == 'UP' and self.direction != (0, 1):
            self.direction = (0, -1)
        elif direction == 'DOWN' and self.direction != (0, -1):
            self.direction = (0, 1)
        elif direction == 'RIGHT' and self.direction != (1, 0):
            self.direction = (-1, 0)
        elif direction == 'LEFT' and self.direction != (-1, 0):
            self.direction = (1, 0)

    def grow(self):
        self.l += 2

    def draw(self):
        for segment in self.body:
            x, y = segment
            pygame.draw.rect(screen, GREEN, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

# Apple class
class Apple:
    def __init__(self):
        self.position = self.generate_position()

    def generate_position(self):
        return random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)

    def draw(self):
        x, y = self.position
        pygame.draw.rect(screen, RED, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))


def main():
    snake = Snake()
    apple = Apple()

    clock = pygame.time.Clock()
    running = True

    while running:
        screen.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        direction = most_common_value(predictions)  # Get the next direction from stored predictions

        print(predictions)

        if direction:
            snake.change_direction(direction)

        if snake.move():
            apple.draw()
            snake.draw()
            if snake.body[0] == apple.position:
                snake.grow()
                apple.position = apple.generate_position()
        else:
            running = False

        pygame.display.flip()
        clock.tick(1)  # Adjust snake speed here

    pygame.quit()


if __name__ == "__main__":
    main()
