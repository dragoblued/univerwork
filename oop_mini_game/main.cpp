#include <GL/freeglut.h> 
#include <cstdlib>

class Painter {
	public:
		enum Color {RED, GREEN, BLUE, YELLOW, MAGENTA, CYAN, PINK, WHITE};
		void rect(int x1, int y1, int x2, int y2);
		void setColor(Color);
};

class Tetramino {
	public:
		enum Direction {LEFT = -1};
		enum Name {I, J, L, O, S, Z, T};
		Tetramino(Name);
		void draw(Painter &) const;
		void move(int dx, int dy);
		void rotate(Direction);
		bool map(int x, int y) const;
		int x() const { return x_; }
		int y() const { return y_; }
	private:
		Name name_;
		int angle_;
		int x_;
		int y_;
};

class Wall {
	public:
		Wall();
		void draw(Painter &) const;
		bool isCollision(const Tetramino &) const;
		int unity(const Tetramino &);
		int removeSolidLines();
	private:
		bool map_[20][10];
};

class Game {
	public:
		enum Direction {UP, DOWN, LEFT, RIGHT};
		Game();
		void draw(Painter &);
		void tick();
		void restart();
		void keyEvent(Direction);
	private:
		Wall wall_;
		Tetramino tetramino_;
};

void Painter::rect(int x1, int y1, int x2, int y2) {
	glBegin(GL_QUADS);
	glVertex2f(x1, y1);
	glVertex2f(x2, y1);
	glVertex2f(x2, y2);
	glVertex2f(x1, y2);
	glEnd();
}

void Painter::setColor(Color color) {
	static const struct {
		float r, g, b;
	}
	colors[] = {
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
		{1, 1, 0},
		{0, 1, 1},
		{1, 0, 1},
		{1, 0.5, 0.8},
		{1, 1, 1},
	};
	glColor3f(colors[color].r, colors[color].g, colors[color].b);
}

Tetramino::Tetramino(Name name): name_(name), angle_(1), x_(3), y_(0) {}

void Tetramino::draw(Painter &p) const {
	p.setColor(static_cast<Painter::Color>(name_));
	for (int y = 0; y < 4; ++y) {
		for (int x = 0; x < 4; ++x) {
			if (map(x, y)) {
				p.rect((x + x_) * 8 + 1, (y + y_) * 8 + 1, (x + x_ + 1) * 8 - 1, (y + y_ + 1) * 8 - 1);
			}
		}
	}
}

bool Tetramino::map(int x, int y) const{
	static const char *SHAPES[] = {
			"  + "
			"  + "
			"  + "
			"  + ",
			 
			"     "
			"   +"
			"   +"
			" ++",

			"     "
			" +  "
			" +  "
			" ++",
			
			" ++"
			" ++"
			"      "
			"      ",
			
			"   +"
			" ++"
			" +  "
			"     ",

			" +  "
			" ++"
			"   +"
			"     ",

			"        "
			" +++" 
			"   +   "
			"        "
	};
	static const struct {
		int x, y;
	}
	ROTARE[][16] = {
		{
			{0, 0}, {0, 1}, {0, 2}, {0, 3},
			{1, 0}, {1, 1}, {1, 2}, {1, 3},
			{2, 0}, {2, 1}, {2, 2}, {2, 3},
			{3, 0}, {3, 1}, {3, 2}, {3, 3}
		},
		{
			{3, 0}, {2, 0}, {1, 0}, {0, 0},
			{3, 1}, {2, 1}, {1, 1}, {0, 1},
			{3, 2}, {2, 2}, {1, 2}, {0, 2},
			{3, 3}, {2, 3}, {1, 3}, {0, 3}
		},
		{
			{3, 3}, {3, 2}, {3, 1}, {3, 0},
			{2, 3}, {2, 2}, {2, 1}, {2, 0},
			{1, 3}, {1, 2}, {1, 1}, {1, 0},
			{0, 3}, {0, 2}, {0, 1}, {0, 0}
		},
		{
			{0, 3}, {1, 3}, {2, 3}, {3, 3},
			{0, 2}, {1, 2}, {2, 2}, {3, 2},
			{0, 1}, {1, 1}, {2, 1}, {3, 1},
			{0, 0}, {1, 0}, {2, 0}, {3, 0}
		}
	};
	return SHAPES[name_] [ROTARE[angle_][y * 4 + x].y * 4 + ROTARE[angle_][y * 4 + x].x] != ' ';
}

void Tetramino::move(int dx, int dy) {
	x_ += dx;
	y_ += dy;
}

void Tetramino::rotate(Direction d) {
	angle_ = (angle_ + d + 4) % 4;
}

Wall::Wall() {
	for (int y = 0; y < 20; ++y) {
		for (int x = 0; x < 10; ++x) {
			map_[y][x] = false;
		}
	}
}

void Wall::draw(Painter &p) const {
	p.setColor(Painter::WHITE);
	for (int y = 0; y < 20; ++y)
		for (int x = 0; x < 10; ++x)
			if (map_[y][x]) {
				p.rect(x * 8 + 1, y * 8 + 1, (x + 1) * 8 - 1, (y + 1) * 8 - 1);
			} else {
				p.rect(x * 8 + 3, y * 8 + 3, (x + 1) * 8 - 4, (y + 1) * 8 - 4); 
			}
}

bool Wall::isCollision(const Tetramino &t) const {
	for (int y = 0; y < 4; ++y) {
		for (int x = 0; x < 4; ++x) {
			if (t.map(x, y)) {
				int wx = x + t.x();
				int wy = y + t.y();
				if (wx < 0 || wx >= 10 || wy < 0 || wy >= 20) {
					return true;
				}
				if (map_[wy][wx]) {
					return true;
				}
			}
		}
	}
	return false;
}

int Wall::unity(const Tetramino &t) {
	for (int y = 0; y < 4; ++y) {
		for (int x = 0; x < 4; ++x) {
			int wx = x + t.x();
			int wy = y + t.y();
			if (wx >= 0 && wx < 10 && wy >= 0 && wy < 20)
				map_[wy][wx] = map_[wy][wx] || t.map(x, y);
		}
	}
}

int Wall::removeSolidLines() {
	int res = 0;
	for (int y = 0; y < 20; ++y) {
		bool solid = true;
		for (int x = 0; x < 10; ++x) {
			if (!map_[y][x]) {
				solid = false;
				break;
			}
		}
		if (solid) {
			++res;
			for (int yy = y - 1; yy >= 0; --yy) {
				for (int x = 0; x < 10; ++x) {
					map_[yy + 1][x] = map_[yy][x];
				}
			}
		}
		for (int x = 0; x < 10; ++x) {
			map_[0][x] = false;
		}
	}
	return res;
}

Game::Game():tetramino_(static_cast<Tetramino::Name>(rand() % 7)){}
	
void Game::draw(Painter &p) {
	wall_.draw(p);
	tetramino_.draw(p);
} 

void Game::tick() {
	Tetramino t = tetramino_;
	t.move(0, 1);
	if (wall_.isCollision(t)) {
		wall_.unity(tetramino_);
		wall_.removeSolidLines();
		tetramino_ = Tetramino(static_cast<Tetramino::Name>(rand() % 7));
		if (wall_.isCollision(tetramino_)) {
			restart();
		}
	} else {
		tetramino_ = t;
	}
}

void Game::restart() {
	wall_ = Wall();
}

void Game::keyEvent(Direction d) {
	Tetramino t = tetramino_;
	switch (d) {
		case UP: 
			t.rotate(Tetramino::LEFT);
			break;
		case DOWN: 
			t.move(0, 1);
			break;
		case LEFT: 
			t.move(-1, 0);
			break;
		case RIGHT: 
			t.move(1, 0);
			break;
	}
	if (!wall_.isCollision(t)) {
		tetramino_ = t;
	}	 
}

Game game;

void display() {
	glClear(GL_COLOR_BUFFER_BIT);
	Painter p;
	game.draw(p);
	glutSwapBuffers();
}

void timer(int) {
	game.tick();
	display();
	glutTimerFunc(1000, timer, 0);
}

void keyEvent(int key, int x, int y) {
	switch (key) {
		case GLUT_KEY_UP:
			game.keyEvent(Game::UP);
			break;
		case GLUT_KEY_DOWN:
			game.keyEvent(Game::DOWN);
			break;
		case GLUT_KEY_LEFT:
			game.keyEvent(Game::LEFT);
			break;
		case GLUT_KEY_RIGHT:
			game.keyEvent(Game::RIGHT);
			break;
	}
	display();
}

int main(int argc, char **argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(10 * 32, 20 * 32);
	glutInitWindowPosition(0, 0);
	glutCreateWindow("Tetris");
	glClearColor(0, 0, 0, 1);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 10 * 8, 20 * 8, 0, -1.0, 1.0);
	glutDisplayFunc(display);
	timer(0);
	glutSpecialFunc(keyEvent);
	glutMainLoop();
	return 0;
}
