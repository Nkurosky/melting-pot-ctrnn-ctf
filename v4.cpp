// V4 single-file toy CTF, translated from v4.py.
//
// This executable contains the complete headless experiment: map geometry,
// collision handling, observations, three-neuron CTRNN controllers, discrete
// actions, pickup-and-return rules, two-population coevolution, elitism,
// Hall-of-Fame opponents, checkpoints, benchmarks, and NumPy-compatible output.
// Visualization remains in v4.py: the .npy genomes written here can be loaded
// there and passed to animate_match.
//
// Build examples:
//   g++ -std=c++17 -O3 -DNDEBUG -static -static-libgcc -static-libstdc++ -o v4.exe v4.cpp
//   cl /std:c++17 /O2 /EHsc v4.cpp /Fe:v4.exe
//
// Small run:
//   .\v4.exe --quick --out-prefix v4_cpp_smoke
// Long run:
//   .\v4.exe --gens 1000 --pop 50 --mut 0.03 --seed 152 --checkpoint-every 50 --benchmark-every 50 --out-prefix v4_cpp_seed152

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ctf {

constexpr double ARENA = 40.0;
constexpr double DT = 0.1;
constexpr int EP_STEPS = 900;
constexpr double AGENT_R = 0.4;
constexpr double CAPTURE_DIST = 1.0;
constexpr double MAX_SPEED = 3.0;
constexpr double WHEEL_BASE = 1.0;
constexpr bool USE_DISCRETE_ACTIONS = true;
constexpr double DISCRETE_STEP = 0.35;
constexpr double PI = 3.14159265358979323846;
constexpr double DISCRETE_TURN = 18.0 * PI / 180.0;

constexpr int MAP_H = 23;
constexpr int MAP_W = 23;
const std::array<std::string, MAP_H> ASCII_MAP = {
    "IIIIIIIIIIIIIIIIIIIIIII",
    "IWWWWWWWWWWWWWWWWWWWWWI",
    "IWPPP,PPPP,F,PPPP,PPPWI",
    "IWPPP,,PP,,,,,PP,,PPPWI",
    "IWPPP,,,,,,,,,,,,,PPPWI",
    "IWP,,WW,,,,,,,,,WW,,PWI",
    "IWHHWWW,WWWWWWW,WWWHHWI",
    "IWHHW,D,,,,,,,,,D,WHHWI",
    "IWHH,,W,,,WWW,,,W,,HHWI",
    "IW,,,,W,,,,,,,,,W,,,,WI",
    "IW,,,,WWW,,,,,WWW,,,,WI",
    "IW,,,,,,,,,I,,,,,,,,,WI",
    "IW,,,,WWW,,,,,WWW,,,,WI",
    "IW,,,,W,,,,,,,,,W,,,,WI",
    "IWHH,,W,,,WWW,,,W,,HHWI",
    "IWHHW,D,,,,,,,,,D,WHHWI",
    "IWHHWWW,WWWWWWW,WWWHHWI",
    "IWQ,,WW,,,,,,,,,WW,,QWI",
    "IWQQQ,,,,,,,,,,,,,QQQWI",
    "IWQQQ,,QQ,,,,,QQ,,QQQWI",
    "IWQQQ,QQQQ,G,QQQQ,QQQWI",
    "IWWWWWWWWWWWWWWWWWWWWWI",
    "IIIIIIIIIIIIIIIIIIIIIII",
};
constexpr double TILE = ARENA / MAP_W;

constexpr int N_SENSORS = 7;
constexpr int N_NEURONS = 3;
constexpr int BASE_PARAMS = N_NEURONS * N_NEURONS + N_NEURONS + N_NEURONS;
constexpr int N_PARAMS = BASE_PARAMS + N_SENSORS * N_NEURONS;
constexpr int OLD_N_PARAMS = N_PARAMS - N_NEURONS;
constexpr double MAX_DIST = 56.568542494923804;

constexpr int TARGET_CAPTURES = 3;
// These two events are the complete fitness function for both teams.
constexpr double FLAG_PICKUP_REWARD = 25.0;
constexpr double FLAG_RETURN_REWARD = 100.0;

struct Vec2 {
    double x = 0.0;
    double y = 0.0;
};

using Genome = std::array<double, N_PARAMS>;
using Sensors = std::array<double, N_SENSORS>;
using Neurons = std::array<double, N_NEURONS>;
using Population = std::vector<Genome>;

enum class Action { None, Noop, Forward, Backward, TurnLeft, TurnRight, Continuous };
enum class Winner { Timeout, Tie, Red, Blue };

const char* action_name(Action action) {
    switch (action) {
        case Action::None: return "none";
        case Action::Noop: return "noop";
        case Action::Forward: return "forward";
        case Action::Backward: return "backward";
        case Action::TurnLeft: return "turn_left";
        case Action::TurnRight: return "turn_right";
        case Action::Continuous: return "continuous";
    }
    return "unknown";
}

const char* winner_name(Winner winner) {
    switch (winner) {
        case Winner::Timeout: return "timeout";
        case Winner::Tie: return "tie";
        case Winner::Red: return "red";
        case Winner::Blue: return "blue";
    }
    return "unknown";
}

double sigmoid(double x) {
    x = std::clamp(x, -30.0, 30.0);
    return 1.0 / (1.0 + std::exp(-x));
}

double wrap_heading(double heading) {
    heading = std::fmod(heading + PI, 2.0 * PI);
    if (heading < 0.0) heading += 2.0 * PI;
    return heading - PI;
}

Vec2 tile_center(int row, int col) {
    return {(col + 0.5) * TILE, ARENA - (row + 0.5) * TILE};
}

char tile_at(double x, double y) {
    const int col = static_cast<int>(std::floor(x / TILE));
    const int row = static_cast<int>(std::floor((ARENA - y) / TILE));
    if (row < 0 || row >= MAP_H || col < 0 || col >= MAP_W) return '\0';
    return ASCII_MAP[row][col];
}

bool is_blocked_tile(char tile) {
    return tile == 'W' || tile == 'H' || tile == 'D';
}

std::vector<Vec2> tile_positions(const std::string& chars) {
    std::vector<Vec2> positions;
    for (int row = 0; row < MAP_H; ++row) {
        for (int col = 0; col < MAP_W; ++col) {
            if (chars.find(ASCII_MAP[row][col]) != std::string::npos) {
                positions.push_back(tile_center(row, col));
            }
        }
    }
    return positions;
}

const std::vector<Vec2> RED_SPAWNS = tile_positions("P");
const std::vector<Vec2> BLUE_SPAWNS = tile_positions("Q");
const Vec2 RED_FLAG_HOME = tile_positions("F").at(0);
const Vec2 BLUE_FLAG_HOME = tile_positions("G").at(0);

bool is_blocked_position(double x, double y, double radius) {
    const std::array<Vec2, 5> samples = {{{x, y}, {x + radius, y},
        {x - radius, y}, {x, y + radius}, {x, y - radius}}};
    for (const Vec2& sample : samples) {
        const char tile = tile_at(sample.x, sample.y);
        if (tile == '\0' || is_blocked_tile(tile)) return true;
    }
    return false;
}

Vec2 move_with_walls(double x, double y, double nx, double ny, double radius) {
    if (!is_blocked_position(nx, ny, radius)) return {nx, ny};
    if (!is_blocked_position(nx, y, radius)) return {nx, y};
    if (!is_blocked_position(x, ny, radius)) return {x, ny};
    return {x, y};
}

class Random {
public:
    explicit Random(std::uint64_t seed) : engine_(seed) {}

    double normal(double mean = 0.0, double stddev = 1.0) {
        return std::normal_distribution<double>(mean, stddev)(engine_);
    }

    int index(int size) {
        if (size <= 0) throw std::runtime_error("cannot sample an empty collection");
        return std::uniform_int_distribution<int>(0, size - 1)(engine_);
    }

    std::vector<int> sample_indices(int size, int count) {
        count = std::min(size, count);
        std::vector<int> indices(size);
        std::iota(indices.begin(), indices.end(), 0);
        for (int i = 0; i < count; ++i) {
            const int j = std::uniform_int_distribution<int>(i, size - 1)(engine_);
            std::swap(indices[i], indices[j]);
        }
        indices.resize(count);
        return indices;
    }

private:
    std::mt19937_64 engine_;
};

Vec2 sample_spawn(Random& rng, const std::vector<Vec2>& positions) {
    return positions.at(rng.index(static_cast<int>(positions.size())));
}

struct Controller {
    std::array<std::array<double, N_NEURONS>, N_NEURONS> recurrent{};
    Neurons bias{};
    Neurons taus{};
    std::array<std::array<double, N_NEURONS>, N_SENSORS> sensor_weights{};
};

Controller unpack_genome(const Genome& genome) {
    Controller controller;
    int i = 0;
    for (int row = 0; row < N_NEURONS; ++row) {
        for (int col = 0; col < N_NEURONS; ++col) {
            controller.recurrent[row][col] = genome[i++];
        }
    }
    for (double& value : controller.bias) value = genome[i++];
    Neurons time_raw{};
    for (double& value : time_raw) value = genome[i++];
    for (int sensor = 0; sensor < N_SENSORS; ++sensor) {
        for (int neuron = 0; neuron < N_NEURONS; ++neuron) {
            controller.sensor_weights[sensor][neuron] = genome[i++];
        }
    }
    for (int neuron = 0; neuron < N_NEURONS; ++neuron) {
        controller.taus[neuron] = 0.5 + 4.5 * sigmoid(time_raw[neuron]);
    }
    return controller;
}

std::array<double, 3> object_state(double ax, double ay, double heading,
                                   double ox, double oy) {
    const double dx = ox - ax;
    const double dy = oy - ay;
    const double forward = dx * std::cos(heading) + dy * std::sin(heading);
    const double left = -dx * std::sin(heading) + dy * std::cos(heading);
    return {forward / ARENA, left / ARENA, std::hypot(dx, dy) / MAX_DIST};
}

Sensors observe_world(double ax, double ay, double heading, double fx, double fy,
                      double px, double py, double last_move_blocked) {
    const auto flag = object_state(ax, ay, heading, fx, fy);
    const auto player = object_state(ax, ay, heading, px, py);
    return {flag[0], flag[1], flag[2], player[0], player[1], player[2],
            last_move_blocked};
}

Action choose_action(const Neurons& output) {
    const double forward_drive = 2.0 * output[1] - 1.0;
    const double turn_drive = 2.0 * output[2] - 1.0;
    if (std::abs(forward_drive) < 0.25 && std::abs(turn_drive) < 0.25) {
        return Action::Noop;
    }
    if (std::abs(forward_drive) >= std::abs(turn_drive)) {
        return forward_drive > 0.0 ? Action::Forward : Action::Backward;
    }
    return turn_drive > 0.0 ? Action::TurnRight : Action::TurnLeft;
}

class Agent {
public:
    Agent(const Genome& genome, double start_x, double start_y, double start_heading)
        : x(start_x), y_pos(start_y), heading(start_heading), genome_(genome),
          controller_(unpack_genome(genome)) {}

    Sensors observe(double fx, double fy, double px, double py) const {
        return observe_world(x, y_pos, heading, fx, fy, px, py, last_move_blocked);
    }

    Sensors step(double fx, double fy, double px, double py) {
        const Sensors obs = observe(fx, fy, px, py);
        Neurons activation{};
        for (int i = 0; i < N_NEURONS; ++i) activation[i] = sigmoid(state_[i] + controller_.bias[i]);

        Neurons next_state = state_;
        for (int neuron = 0; neuron < N_NEURONS; ++neuron) {
            double recurrent_current = 0.0;
            for (int input = 0; input < N_NEURONS; ++input) {
                recurrent_current += controller_.recurrent[neuron][input] * activation[input];
            }
            double sensor_current = 0.0;
            for (int sensor = 0; sensor < N_SENSORS; ++sensor) {
                sensor_current += controller_.sensor_weights[sensor][neuron] * obs[sensor];
            }
            const double dy = (-state_[neuron] + recurrent_current + sensor_current)
                            / controller_.taus[neuron];
            next_state[neuron] += DT * dy;
        }
        state_ = next_state;

        Neurons output{};
        for (int i = 0; i < N_NEURONS; ++i) output[i] = sigmoid(state_[i] + controller_.bias[i]);

        if (USE_DISCRETE_ACTIONS) {
            last_action = choose_action(output);
            apply_discrete_action(last_action);
            return obs;
        }

        last_action = Action::Continuous;
        vl = (2.0 * output[1] - 1.0) * MAX_SPEED;
        vr = (2.0 * output[2] - 1.0) * MAX_SPEED;
        const double v = 0.5 * (vl + vr);
        const double w = (vr - vl) / WHEEL_BASE;
        heading = wrap_heading(heading + w * DT);
        const double nx = std::clamp(x + v * std::cos(heading) * DT, AGENT_R, ARENA - AGENT_R);
        const double ny = std::clamp(y_pos + v * std::sin(heading) * DT, AGENT_R, ARENA - AGENT_R);
        const double old_x = x;
        const double old_y = y_pos;
        const Vec2 moved = move_with_walls(x, y_pos, nx, ny, AGENT_R);
        x = moved.x;
        y_pos = moved.y;
        const double intended = std::hypot(nx - old_x, ny - old_y);
        const double actual = std::hypot(x - old_x, y_pos - old_y);
        last_move_blocked = intended > 1e-6 && actual < 1e-6 ? 1.0 : 0.0;
        return obs;
    }

    double x = 0.0;
    double y_pos = 0.0;
    double heading = 0.0;
    double vl = 0.0;
    double vr = 0.0;
    Action last_action = Action::None;
    double last_move_blocked = 0.0;

private:
    void apply_discrete_action(Action action) {
        vl = 0.0;
        vr = 0.0;
        last_move_blocked = 0.0;
        if (action == Action::TurnLeft) {
            heading -= DISCRETE_TURN;
        } else if (action == Action::TurnRight) {
            heading += DISCRETE_TURN;
        } else if (action == Action::Forward || action == Action::Backward) {
            const double direction = action == Action::Forward ? 1.0 : -1.0;
            const double nx = std::clamp(x + direction * DISCRETE_STEP * std::cos(heading),
                                         AGENT_R, ARENA - AGENT_R);
            const double ny = std::clamp(y_pos + direction * DISCRETE_STEP * std::sin(heading),
                                         AGENT_R, ARENA - AGENT_R);
            const double old_x = x;
            const double old_y = y_pos;
            const Vec2 moved = move_with_walls(x, y_pos, nx, ny, AGENT_R);
            x = moved.x;
            y_pos = moved.y;
            last_move_blocked = std::hypot(x - old_x, y_pos - old_y) < 1e-6 ? 1.0 : 0.0;
        } else if (action != Action::Noop) {
            throw std::runtime_error("unknown discrete action");
        }
        heading = wrap_heading(heading);
    }

    Genome genome_{};
    Controller controller_{};
    Neurons state_{};
};

Vec2 home_flag(int team) { return team == 0 ? RED_FLAG_HOME : BLUE_FLAG_HOME; }
Vec2 enemy_flag(int team) { return team == 0 ? BLUE_FLAG_HOME : RED_FLAG_HOME; }

Agent spawn_agent(Random& rng, int team, const Genome& genome) {
    const Vec2 spawn = sample_spawn(rng, team == 0 ? RED_SPAWNS : BLUE_SPAWNS);
    return Agent(genome, spawn.x, spawn.y, team == 0 ? -PI / 2.0 : PI / 2.0);
}

double distance_to(const Agent& agent, Vec2 point) {
    return std::hypot(agent.x - point.x, agent.y_pos - point.y);
}

struct Frame {
    int t = 0;
    double red_x = 0.0, red_y = 0.0, red_heading = 0.0;
    Action red_action = Action::None;
    double blue_x = 0.0, blue_y = 0.0, blue_heading = 0.0;
    Action blue_action = Action::None;
    std::array<int, 2> captures{};
    std::array<bool, 2> carrying{};
    Sensors red_obs{};
    Sensors blue_obs{};
};

struct MatchResult {
    std::array<double, 2> fitness{};
    Winner winner = Winner::Timeout;
    std::array<int, 2> captures{};
    std::vector<Frame> trail;
};

MatchResult run_match(const Genome& red_genome, const Genome& blue_genome,
                      std::uint64_t seed, bool record = false) {
    Random rng(seed);
    Agent red = spawn_agent(rng, 0, red_genome);
    Agent blue = spawn_agent(rng, 1, blue_genome);
    std::array<bool, 2> carrying{};
    MatchResult result;
    if (record) result.trail.reserve(EP_STEPS);

    for (int t = 0; t < EP_STEPS; ++t) {
        const Vec2 red_target = carrying[0] ? home_flag(0) : enemy_flag(0);
        const Vec2 blue_target = carrying[1] ? home_flag(1) : enemy_flag(1);
        const Sensors red_obs = red.step(red_target.x, red_target.y, blue.x, blue.y_pos);
        const Sensors blue_obs = blue.step(blue_target.x, blue_target.y, red.x, red.y_pos);
        std::array<Agent*, 2> agents = {&red, &blue};

        for (int team = 0; team < 2; ++team) {
            Agent& agent = *agents[team];
            if (!carrying[team] && distance_to(agent, enemy_flag(team)) < CAPTURE_DIST) {
                carrying[team] = true;
                result.fitness[team] += FLAG_PICKUP_REWARD;
            } else if (carrying[team] && distance_to(agent, home_flag(team)) < CAPTURE_DIST) {
                ++result.captures[team];
                carrying[team] = false;
                result.fitness[team] += FLAG_RETURN_REWARD;
            }
        }

        if (record) {
            result.trail.push_back({t, red.x, red.y_pos, red.heading, red.last_action,
                blue.x, blue.y_pos, blue.heading, blue.last_action,
                result.captures, carrying, red_obs, blue_obs});
        }

        if (result.captures[0] >= TARGET_CAPTURES && result.captures[1] >= TARGET_CAPTURES) {
            result.winner = Winner::Tie;
            break;
        }
        if (result.captures[0] >= TARGET_CAPTURES) {
            result.winner = Winner::Red;
            break;
        }
        if (result.captures[1] >= TARGET_CAPTURES) {
            result.winner = Winner::Blue;
            break;
        }
    }
    return result;
}

// Minimal NumPy .npy v1.0 support for little-endian float64 arrays.
std::vector<double> load_npy(const std::string& path, std::vector<std::size_t>& shape) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("could not open " + path);
    char magic[6]{};
    input.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error(path + " is not an NPY file");
    unsigned char version[2]{};
    input.read(reinterpret_cast<char*>(version), 2);
    std::uint32_t header_length = 0;
    if (version[0] == 1) {
        std::uint16_t length16 = 0;
        input.read(reinterpret_cast<char*>(&length16), 2);
        header_length = length16;
    } else {
        input.read(reinterpret_cast<char*>(&header_length), 4);
    }
    std::string header(header_length, '\0');
    input.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (header.find("f8") == std::string::npos || header.find("True") != std::string::npos) {
        throw std::runtime_error(path + " must be a C-order float64 NPY array");
    }
    const auto open = header.find('(');
    const auto close = header.find(')', open);
    if (open == std::string::npos || close == std::string::npos) throw std::runtime_error("invalid NPY shape");
    std::stringstream dimensions(header.substr(open + 1, close - open - 1));
    std::string token;
    while (std::getline(dimensions, token, ',')) {
        std::stringstream value(token);
        std::size_t dimension = 0;
        if (value >> dimension) shape.push_back(dimension);
    }
    const std::size_t count = std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                                               std::multiplies<std::size_t>());
    std::vector<double> data(count);
    input.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count * sizeof(double)));
    if (!input) throw std::runtime_error("truncated data in " + path);
    return data;
}

void save_npy(const std::string& path, const std::vector<double>& data,
              const std::vector<std::size_t>& shape) {
    const std::size_t expected = std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                                                  std::multiplies<std::size_t>());
    if (expected != data.size()) throw std::runtime_error("NPY shape does not match data");
    std::ostringstream shape_text;
    shape_text << '(';
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i) shape_text << ", ";
        shape_text << shape[i];
    }
    if (shape.size() == 1) shape_text << ',';
    shape_text << ')';
    std::string header = "{'descr': '<f8', 'fortran_order': False, 'shape': " + shape_text.str() + ", }";
    const std::size_t preamble = 10;
    const std::size_t padding = (16 - ((preamble + header.size() + 1) % 16)) % 16;
    header.append(padding, ' ');
    header.push_back('\n');
    const std::uint16_t header_length = static_cast<std::uint16_t>(header.size());
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("could not write " + path);
    output.write("\x93NUMPY", 6);
    const unsigned char version[2] = {1, 0};
    output.write(reinterpret_cast<const char*>(version), 2);
    output.write(reinterpret_cast<const char*>(&header_length), 2);
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write(reinterpret_cast<const char*>(data.data()),
                 static_cast<std::streamsize>(data.size() * sizeof(double)));
}

Genome load_genome(const std::string& path) {
    std::vector<std::size_t> shape;
    const std::vector<double> data = load_npy(path, shape);
    Genome genome{};
    if (data.size() == N_PARAMS) {
        std::copy(data.begin(), data.end(), genome.begin());
    } else if (data.size() == OLD_N_PARAMS) {
        std::copy(data.begin(), data.end(), genome.begin());
        std::fill(genome.begin() + OLD_N_PARAMS, genome.end(), 0.0);
    } else {
        throw std::runtime_error("genome in " + path + " has " + std::to_string(data.size())
                                 + " values; expected 33 or 36");
    }
    return genome;
}

void save_genome(const std::string& path, const Genome& genome) {
    save_npy(path, std::vector<double>(genome.begin(), genome.end()), {N_PARAMS});
}

void save_population(const std::string& path, const Population& population) {
    std::vector<double> data;
    data.reserve(population.size() * N_PARAMS);
    for (const Genome& genome : population) data.insert(data.end(), genome.begin(), genome.end());
    save_npy(path, data, {population.size(), N_PARAMS});
}

Population init_population(Random& rng, int pop, const std::string& resume = "",
                           double jitter = 0.08) {
    Population genomes(pop);
    if (!resume.empty()) {
        const Genome seed_genome = load_genome(resume);
        for (Genome& genome : genomes) {
            for (int p = 0; p < N_PARAMS; ++p) genome[p] = seed_genome[p] + rng.normal(0.0, jitter);
        }
        genomes[0] = seed_genome;
    } else {
        for (Genome& genome : genomes) for (double& value : genome) value = rng.normal();
    }
    return genomes;
}

struct Evaluation {
    std::vector<double> red_fits, blue_fits;
    std::vector<double> red_winrate, blue_winrate;
    std::vector<double> red_caprate, blue_caprate;
};

Evaluation evaluate(const Population& red_genomes, const Population& blue_genomes,
                    const Population& red_hof, const Population& blue_hof,
                    int n_opponents = 4, int n_hof = 2, std::uint64_t seed = 0) {
    Random rng(seed);
    const int red_pop = static_cast<int>(red_genomes.size());
    const int blue_pop = static_cast<int>(blue_genomes.size());
    Evaluation eval;
    eval.red_fits.assign(red_pop, 0.0); eval.blue_fits.assign(blue_pop, 0.0);
    std::vector<double> red_wins(red_pop), blue_wins(blue_pop), red_caps(red_pop),
                        blue_caps(blue_pop), red_matches(red_pop), blue_matches(blue_pop);

    for (int i = 0; i < red_pop; ++i) {
        struct Opponent { bool from_population; int index; const Genome* genome; };
        std::vector<Opponent> opponents;
        for (int index : rng.sample_indices(blue_pop, n_opponents)) {
            opponents.push_back({true, index, &blue_genomes[index]});
        }
        for (int index : rng.sample_indices(static_cast<int>(blue_hof.size()), n_hof)) {
            opponents.push_back({false, index, &blue_hof[index]});
        }
        for (int j = 0; j < static_cast<int>(opponents.size()); ++j) {
            const std::uint64_t match_seed = (seed * 1000003ULL + i * 1009ULL + j * 17ULL) & 0xFFFFFFFFULL;
            const MatchResult match = run_match(red_genomes[i], *opponents[j].genome, match_seed);
            eval.red_fits[i] += match.fitness[0];
            red_caps[i] += match.captures[0];
            red_matches[i] += 1.0;
            if (match.winner == Winner::Red) red_wins[i] += 1.0;
            if (opponents[j].from_population) {
                const int index = opponents[j].index;
                eval.blue_fits[index] += match.fitness[1];
                blue_caps[index] += match.captures[1];
                blue_matches[index] += 1.0;
                if (match.winner == Winner::Blue) blue_wins[index] += 1.0;
            }
        }
    }

    for (int i = 0; i < blue_pop; ++i) {
        if (blue_matches[i] > 0.0) continue;
        std::vector<const Genome*> opponents;
        for (int index : rng.sample_indices(red_pop, n_opponents)) opponents.push_back(&red_genomes[index]);
        for (int index : rng.sample_indices(static_cast<int>(red_hof.size()), n_hof)) opponents.push_back(&red_hof[index]);
        for (int j = 0; j < static_cast<int>(opponents.size()); ++j) {
            const std::uint64_t match_seed = (seed * 2000003ULL + i * 1013ULL + j * 19ULL) & 0xFFFFFFFFULL;
            const MatchResult match = run_match(*opponents[j], blue_genomes[i], match_seed);
            eval.blue_fits[i] += match.fitness[1];
            blue_caps[i] += match.captures[1];
            blue_matches[i] += 1.0;
            if (match.winner == Winner::Blue) blue_wins[i] += 1.0;
        }
    }

    eval.red_winrate.resize(red_pop); eval.blue_winrate.resize(blue_pop);
    eval.red_caprate.resize(red_pop); eval.blue_caprate.resize(blue_pop);
    for (int i = 0; i < red_pop; ++i) {
        const double matches = std::max(red_matches[i], 1.0);
        eval.red_fits[i] /= matches;
        eval.red_winrate[i] = red_wins[i] / matches;
        eval.red_caprate[i] = red_caps[i] / matches;
    }
    for (int i = 0; i < blue_pop; ++i) {
        const double matches = std::max(blue_matches[i], 1.0);
        eval.blue_fits[i] /= matches;
        eval.blue_winrate[i] = blue_wins[i] / matches;
        eval.blue_caprate[i] = blue_caps[i] / matches;
    }
    return eval;
}

Population next_generation(Random& rng, const Population& genomes,
                           const std::vector<double>& fits, double mutation, int elite) {
    std::vector<int> order(genomes.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return fits[a] > fits[b]; });
    Population ranked;
    ranked.reserve(genomes.size());
    for (int index : order) ranked.push_back(genomes[index]);

    Population next;
    next.reserve(genomes.size());
    for (int i = 0; i < std::min<int>(elite, ranked.size()); ++i) next.push_back(ranked[i]);
    while (next.size() < ranked.size()) {
        const std::vector<int> tournament = rng.sample_indices(static_cast<int>(ranked.size()),
                                                               std::min<int>(3, ranked.size()));
        const int parent_index = *std::min_element(tournament.begin(), tournament.end());
        Genome child = ranked[parent_index];
        for (double& value : child) value += rng.normal(0.0, mutation);
        next.push_back(child);
    }
    return next;
}

struct Benchmark {
    double red_winrate = 0.0, blue_winrate = 0.0, tie_rate = 0.0, timeout_rate = 0.0;
    double red_capture_rate = 0.0, blue_capture_rate = 0.0;
    double red_mean_captures = 0.0, blue_mean_captures = 0.0;
};

Benchmark benchmark_pair(const Genome& red, const Genome& blue, int first_seed, int trials) {
    int red_wins = 0, blue_wins = 0, ties = 0, timeouts = 0, red_any = 0, blue_any = 0;
    int red_captures = 0, blue_captures = 0;
    for (int i = 0; i < trials; ++i) {
        const MatchResult match = run_match(red, blue, static_cast<std::uint64_t>(first_seed + i));
        red_captures += match.captures[0]; blue_captures += match.captures[1];
        red_any += match.captures[0] > 0; blue_any += match.captures[1] > 0;
        red_wins += match.winner == Winner::Red; blue_wins += match.winner == Winner::Blue;
        ties += match.winner == Winner::Tie; timeouts += match.winner == Winner::Timeout;
    }
    const double n = std::max(trials, 1);
    return {100.0 * red_wins / n, 100.0 * blue_wins / n, 100.0 * ties / n,
            100.0 * timeouts / n, 100.0 * red_any / n, 100.0 * blue_any / n,
            red_captures / n, blue_captures / n};
}

struct Outcome { std::string winner; int red_captures = 0; int blue_captures = 0; };
struct BenchmarkRow { int generation = 0; Benchmark values; };

struct EvolutionResult {
    Genome best_red{}, best_blue{};
    double best_red_fit = -std::numeric_limits<double>::infinity();
    double best_blue_fit = -std::numeric_limits<double>::infinity();
    Population red_hof, blue_hof;
    std::vector<double> red_best_hist, blue_best_hist, red_mean_hist, blue_mean_hist;
    std::vector<BenchmarkRow> benchmark_hist;
    std::vector<Outcome> outcome_hist;
};

void save_outputs(const std::string& prefix, const EvolutionResult& result, int checkpoint = -1) {
    std::ostringstream suffix;
    if (checkpoint >= 0) suffix << "_gen" << std::setw(4) << std::setfill('0') << checkpoint;
    save_genome(prefix + suffix.str() + "_red.npy", result.best_red);
    save_genome(prefix + suffix.str() + "_blue.npy", result.best_blue);
    save_population(prefix + "_red_hof.npy", result.red_hof);
    save_population(prefix + "_blue_hof.npy", result.blue_hof);
    save_npy(prefix + "_red_best_hist.npy", result.red_best_hist, {result.red_best_hist.size()});
    save_npy(prefix + "_blue_best_hist.npy", result.blue_best_hist, {result.blue_best_hist.size()});
    save_npy(prefix + "_red_mean_hist.npy", result.red_mean_hist, {result.red_mean_hist.size()});
    save_npy(prefix + "_blue_mean_hist.npy", result.blue_mean_hist, {result.blue_mean_hist.size()});

    std::vector<double> benchmark_data;
    for (const BenchmarkRow& row : result.benchmark_hist) {
        const Benchmark& b = row.values;
        const std::array<double, 8> values = {static_cast<double>(row.generation), b.red_winrate,
            b.blue_winrate, b.red_capture_rate, b.blue_capture_rate,
            b.red_mean_captures, b.blue_mean_captures, b.timeout_rate};
        benchmark_data.insert(benchmark_data.end(), values.begin(), values.end());
    }
    save_npy(prefix + "_benchmark.npy", benchmark_data, {result.benchmark_hist.size(), 8});

    std::ofstream outcomes(prefix + "_outcomes.csv");
    outcomes << "generation,winner,red_captures,blue_captures\n";
    for (std::size_t i = 0; i < result.outcome_hist.size(); ++i) {
        const Outcome& outcome = result.outcome_hist[i];
        outcomes << i << ',' << outcome.winner << ',' << outcome.red_captures << ','
                 << outcome.blue_captures << '\n';
    }
    if (checkpoint >= 0) {
        std::cout << "checkpoint saved at generation " << checkpoint << ": "
                  << prefix << suffix.str() << "_*.npy\n";
    }
}

struct EvolutionOptions {
    int pop = 50;
    int gens = 1000;
    double mutation = 0.05;
    int elite = 2;
    std::uint64_t seed = 0;
    std::string red_resume, blue_resume;
    int n_opponents = 4;
    int hof_every = 10;
    int hof_cap = 12;
    int checkpoint_every = 0;
    int benchmark_every = 0;
    int benchmark_trials = 100;
    double max_hours = 0.0;
    std::string out_prefix = "v4_cpp";
};

EvolutionResult evolve(const EvolutionOptions& options) {
    Random rng(options.seed);
    Population red = init_population(rng, options.pop, options.red_resume);
    Population blue = init_population(rng, options.pop, options.blue_resume);
    if (!options.red_resume.empty()) std::cout << "Red resumed from " << options.red_resume << '\n';
    if (!options.blue_resume.empty()) std::cout << "Blue resumed from " << options.blue_resume << '\n';
    EvolutionResult result;
    const auto training_started = std::chrono::steady_clock::now();

    for (int generation = 0; generation < options.gens; ++generation) {
        Evaluation eval = evaluate(red, blue, result.red_hof, result.blue_hof,
                                   options.n_opponents, 2, generation + 1);
        const int red_top = static_cast<int>(std::max_element(eval.red_fits.begin(), eval.red_fits.end())
                                             - eval.red_fits.begin());
        const int blue_top = static_cast<int>(std::max_element(eval.blue_fits.begin(), eval.blue_fits.end())
                                              - eval.blue_fits.begin());
        result.red_best_hist.push_back(eval.red_fits[red_top]);
        result.blue_best_hist.push_back(eval.blue_fits[blue_top]);
        result.red_mean_hist.push_back(std::accumulate(eval.red_fits.begin(), eval.red_fits.end(), 0.0)
                                       / eval.red_fits.size());
        result.blue_mean_hist.push_back(std::accumulate(eval.blue_fits.begin(), eval.blue_fits.end(), 0.0)
                                        / eval.blue_fits.size());
        if (eval.red_fits[red_top] > result.best_red_fit) {
            result.best_red_fit = eval.red_fits[red_top]; result.best_red = red[red_top];
        }
        if (eval.blue_fits[blue_top] > result.best_blue_fit) {
            result.best_blue_fit = eval.blue_fits[blue_top]; result.best_blue = blue[blue_top];
        }

        const MatchResult top_match = run_match(red[red_top], blue[blue_top], generation * 131ULL + 7ULL);
        result.outcome_hist.push_back({winner_name(top_match.winner), top_match.captures[0],
                                      top_match.captures[1]});
        std::cout << "gen " << std::setw(4) << generation << std::fixed << std::setprecision(1)
                  << "  red_best=" << std::setw(7) << eval.red_fits[red_top]
                  << " wr=" << std::setprecision(2) << eval.red_winrate[red_top]
                  << " cap=" << eval.red_caprate[red_top]
                  << std::setprecision(1) << "  blue_best=" << std::setw(7) << eval.blue_fits[blue_top]
                  << " wr=" << std::setprecision(2) << eval.blue_winrate[blue_top]
                  << " cap=" << eval.blue_caprate[blue_top]
                  << "  top_match=" << winner_name(top_match.winner)
                  << " caps=[" << top_match.captures[0] << ", " << top_match.captures[1] << "]"
                  << " hof=(" << result.red_hof.size() << ',' << result.blue_hof.size() << ")\n";

        if (options.benchmark_every > 0
            && (generation % options.benchmark_every == 0 || generation == options.gens - 1)) {
            const Benchmark bench = benchmark_pair(red[red_top], blue[blue_top], 900000,
                                                   options.benchmark_trials);
            result.benchmark_hist.push_back({generation, bench});
            std::cout << "          benchmark" << options.benchmark_trials << std::setprecision(1)
                      << " red_win=" << std::setw(5) << bench.red_winrate
                      << " blue_win=" << std::setw(5) << bench.blue_winrate
                      << " red_cap=" << std::setw(5) << bench.red_capture_rate
                      << " blue_cap=" << std::setw(5) << bench.blue_capture_rate
                      << " timeout=" << std::setw(5) << bench.timeout_rate << '\n';
        }

        if (generation % options.hof_every == 0) {
            result.red_hof.push_back(red[red_top]); result.blue_hof.push_back(blue[blue_top]);
            if (static_cast<int>(result.red_hof.size()) > options.hof_cap) result.red_hof.erase(result.red_hof.begin());
            if (static_cast<int>(result.blue_hof.size()) > options.hof_cap) result.blue_hof.erase(result.blue_hof.begin());
        }
        if (options.checkpoint_every > 0
            && ((generation + 1) % options.checkpoint_every == 0 || generation == options.gens - 1)) {
            save_outputs(options.out_prefix, result, generation + 1);
        }
        if (options.max_hours > 0.0) {
            const std::chrono::duration<double, std::ratio<3600>> elapsed =
                std::chrono::steady_clock::now() - training_started;
            if (elapsed.count() >= options.max_hours) {
                save_outputs(options.out_prefix, result, generation + 1);
                std::cout << "wall-clock limit reached after " << generation + 1
                          << " generations (" << std::setprecision(3) << elapsed.count()
                          << " hours)\n";
                break;
            }
        }
        red = next_generation(rng, red, eval.red_fits, options.mutation, options.elite);
        blue = next_generation(rng, blue, eval.blue_fits, options.mutation, options.elite);
    }
    return result;
}

void save_replay_csv(const std::string& path, const MatchResult& match) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("could not write " + path);
    output << "t,red_x,red_y,red_heading,red_action,blue_x,blue_y,blue_heading,blue_action,"
              "red_captures,blue_captures,red_carrying,blue_carrying,"
              "red_flag_fwd,red_flag_left,red_flag_dist,red_player_fwd,red_player_left,red_player_dist,red_blocked,"
              "blue_flag_fwd,blue_flag_left,blue_flag_dist,blue_player_fwd,blue_player_left,blue_player_dist,blue_blocked\n";
    output << std::setprecision(17);
    for (const Frame& frame : match.trail) {
        output << frame.t << ',' << frame.red_x << ',' << frame.red_y << ',' << frame.red_heading << ','
               << action_name(frame.red_action) << ',' << frame.blue_x << ',' << frame.blue_y << ','
               << frame.blue_heading << ',' << action_name(frame.blue_action) << ','
               << frame.captures[0] << ',' << frame.captures[1] << ',' << frame.carrying[0] << ','
               << frame.carrying[1];
        for (double value : frame.red_obs) output << ',' << value;
        for (double value : frame.blue_obs) output << ',' << value;
        output << '\n';
    }
}

void save_fitness_svg(const std::string& path, const EvolutionResult& result) {
    constexpr double width = 900.0;
    constexpr double height = 500.0;
    constexpr double left = 75.0;
    constexpr double right = 25.0;
    constexpr double top = 45.0;
    constexpr double bottom = 60.0;
    const std::array<const std::vector<double>*, 4> series = {
        &result.red_best_hist, &result.blue_best_hist,
        &result.red_mean_hist, &result.blue_mean_hist,
    };
    double low = std::numeric_limits<double>::infinity();
    double high = -std::numeric_limits<double>::infinity();
    std::size_t points = 0;
    for (const auto* values : series) {
        points = std::max(points, values->size());
        for (double value : *values) {
            low = std::min(low, value);
            high = std::max(high, value);
        }
    }
    if (points == 0) return;
    if (std::abs(high - low) < 1e-12) { low -= 1.0; high += 1.0; }
    const double plot_width = width - left - right;
    const double plot_height = height - top - bottom;
    auto sx = [&](std::size_t i) {
        return left + (points <= 1 ? 0.0 : static_cast<double>(i) / (points - 1) * plot_width);
    };
    auto sy = [&](double value) {
        return top + (high - value) / (high - low) * plot_height;
    };

    std::ofstream output(path);
    if (!output) throw std::runtime_error("could not write " + path);
    output << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"900\" height=\"500\" viewBox=\"0 0 900 500\">\n"
              "<rect width=\"900\" height=\"500\" fill=\"white\"/>\n"
              "<text x=\"450\" y=\"27\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"18\">v4 pickup-return CTF</text>\n"
              "<line x1=\"75\" y1=\"440\" x2=\"875\" y2=\"440\" stroke=\"#222\"/>\n"
              "<line x1=\"75\" y1=\"45\" x2=\"75\" y2=\"440\" stroke=\"#222\"/>\n";
    output << "<text x=\"475\" y=\"485\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"14\">generation</text>\n"
              "<text x=\"18\" y=\"245\" text-anchor=\"middle\" transform=\"rotate(-90 18 245)\" font-family=\"sans-serif\" font-size=\"14\">pickup-return fitness</text>\n";
    for (int tick = 0; tick <= 5; ++tick) {
        const double value = low + (high - low) * tick / 5.0;
        const double y = sy(value);
        output << "<line x1=\"75\" y1=\"" << y << "\" x2=\"875\" y2=\"" << y
               << "\" stroke=\"#dddddd\"/>\n<text x=\"68\" y=\"" << y + 5
               << "\" text-anchor=\"end\" font-family=\"sans-serif\" font-size=\"12\">"
               << std::fixed << std::setprecision(1) << value << "</text>\n";
    }
    const std::array<std::string, 4> colors = {"#d62728", "#1f77b4", "#e58c8c", "#82add0"};
    const std::array<std::string, 4> labels = {"red best", "blue best", "red mean", "blue mean"};
    for (int line = 0; line < 4; ++line) {
        output << "<polyline fill=\"none\" stroke=\"" << colors[line]
               << "\" stroke-width=\"" << (line < 2 ? 2.0 : 1.4) << "\" points=\"";
        for (std::size_t i = 0; i < series[line]->size(); ++i) {
            output << sx(i) << ',' << sy((*series[line])[i]) << ' ';
        }
        output << "\"/>\n";
        const double legend_x = 600.0 + (line % 2) * 130.0;
        const double legend_y = 60.0 + (line / 2) * 22.0;
        output << "<line x1=\"" << legend_x << "\" y1=\"" << legend_y << "\" x2=\""
               << legend_x + 25 << "\" y2=\"" << legend_y << "\" stroke=\"" << colors[line]
               << "\" stroke-width=\"2\"/><text x=\"" << legend_x + 31 << "\" y=\""
               << legend_y + 4 << "\" font-family=\"sans-serif\" font-size=\"12\">"
               << labels[line] << "</text>\n";
    }
    output << "</svg>\n";
}

struct CommandLine {
    EvolutionOptions evolution;
    bool quick = false;
    bool replay_only = false;
    std::string replay_red, replay_blue, replay_out = "v4_cpp_replay.csv";
    std::uint64_t replay_seed = 42;
};

void print_help() {
    std::cout <<
        "V4 pickup-and-return CTF (C++17)\n\n"
        "Training:\n"
        "  v4.exe [--quick] [--pop N] [--gens N] [--mut X] [--elite N]\n"
        "         [--seed N] [--red-resume FILE] [--blue-resume FILE]\n"
        "         [--n-opponents N] [--checkpoint-every N]\n"
        "         [--benchmark-every N] [--benchmark-trials N]\n"
        "         [--max-hours H] [--out-prefix NAME]\n"
        "\n"
        "Replay to CSV:\n"
        "  v4.exe --replay RED.npy BLUE.npy [--replay-seed N] [--replay-out FILE.csv]\n";
}

CommandLine parse_args(int argc, char** argv) {
    CommandLine command;
    bool pop_set = false, gens_set = false;
    auto require_value = [&](int& i, const std::string& option) -> std::string {
        if (++i >= argc) throw std::runtime_error(option + " requires a value");
        return argv[i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--help" || option == "-h") { print_help(); std::exit(0); }
        else if (option == "--quick") command.quick = true;
        else if (option == "--pop") { command.evolution.pop = std::stoi(require_value(i, option)); pop_set = true; }
        else if (option == "--gens") { command.evolution.gens = std::stoi(require_value(i, option)); gens_set = true; }
        else if (option == "--mut") command.evolution.mutation = std::stod(require_value(i, option));
        else if (option == "--elite") command.evolution.elite = std::stoi(require_value(i, option));
        else if (option == "--seed") command.evolution.seed = std::stoull(require_value(i, option));
        else if (option == "--red-resume") command.evolution.red_resume = require_value(i, option);
        else if (option == "--blue-resume") command.evolution.blue_resume = require_value(i, option);
        else if (option == "--n-opponents") command.evolution.n_opponents = std::stoi(require_value(i, option));
        else if (option == "--checkpoint-every") command.evolution.checkpoint_every = std::stoi(require_value(i, option));
        else if (option == "--benchmark-every") command.evolution.benchmark_every = std::stoi(require_value(i, option));
        else if (option == "--benchmark-trials") command.evolution.benchmark_trials = std::stoi(require_value(i, option));
        else if (option == "--max-hours") command.evolution.max_hours = std::stod(require_value(i, option));
        else if (option == "--out-prefix") command.evolution.out_prefix = require_value(i, option);
        else if (option == "--replay") {
            command.replay_only = true;
            command.replay_red = require_value(i, option);
            command.replay_blue = require_value(i, option);
        }
        else if (option == "--replay-seed") command.replay_seed = std::stoull(require_value(i, option));
        else if (option == "--replay-out") command.replay_out = require_value(i, option);
        else if (option == "--no-animate") { /* Accepted for command compatibility. */ }
        else throw std::runtime_error("unknown option: " + option);
    }
    if (command.quick) {
        if (!pop_set) command.evolution.pop = 12;
        if (!gens_set) command.evolution.gens = 10;
    }
    if (command.evolution.pop <= 0 || command.evolution.gens <= 0) throw std::runtime_error("pop and gens must be positive");
    if (command.evolution.elite < 0 || command.evolution.elite > command.evolution.pop) throw std::runtime_error("elite must be between 0 and pop");
    if (command.evolution.max_hours < 0.0) throw std::runtime_error("max-hours cannot be negative");
    return command;
}

}  // namespace ctf

int main(int argc, char** argv) {
    try {
        const ctf::CommandLine command = ctf::parse_args(argc, argv);
        if (command.replay_only) {
            const ctf::Genome red = ctf::load_genome(command.replay_red);
            const ctf::Genome blue = ctf::load_genome(command.replay_blue);
            const ctf::MatchResult match = ctf::run_match(red, blue, command.replay_seed, true);
            ctf::save_replay_csv(command.replay_out, match);
            std::cout << "winner=" << ctf::winner_name(match.winner)
                      << " red_fit=" << match.fitness[0] << " blue_fit=" << match.fitness[1]
                      << " captures=[" << match.captures[0] << ", " << match.captures[1] << "]\n"
                      << "Saved replay to " << command.replay_out << '\n';
            return 0;
        }
        const ctf::EvolutionResult result = ctf::evolve(command.evolution);
        ctf::save_outputs(command.evolution.out_prefix, result);
        ctf::save_fitness_svg(command.evolution.out_prefix + "_fitness.svg", result);
        std::cout << "Saved " << command.evolution.out_prefix << "_red.npy and "
                  << command.evolution.out_prefix << "_blue.npy (best fits: red="
                  << result.best_red_fit << ", blue=" << result.best_blue_fit << ")\n"
                  << "Saved " << command.evolution.out_prefix << "_fitness.svg\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
