// V5: 4v4 shared-base CTRNN CTF with persistent variants and Elo league play.
// Build: g++ -std=c++17 -O3 -DNDEBUG -Wall -Wextra -Wpedantic -static -static-libgcc -static-libstdc++ -o v5_elo_4v4.exe v5_elo_4v4.cpp

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

namespace league {

constexpr double ARENA = 40.0, DT = 0.1, AGENT_R = 0.4, CAPTURE_DIST = 1.0;
constexpr double DISCRETE_STEP = 0.35, PI = 3.14159265358979323846;
constexpr double DISCRETE_TURN = 18.0 * PI / 180.0;
constexpr double MAX_DIST = 56.568542494923804;
constexpr int EP_STEPS = 900, TARGET_CAPTURES = 3, TEAM_SIZE = 4, AGENT_COUNT = 8;
constexpr int MAP_H = 23, MAP_W = 23;
constexpr double TILE = ARENA / MAP_W;

constexpr int N_NEURONS = 3, N_SENSORS = 10;
constexpr int BASE_PARAMS = N_NEURONS * N_NEURONS + 2 * N_NEURONS;
constexpr int N_PARAMS = BASE_PARAMS + N_SENSORS * N_NEURONS;  // 45
constexpr int V4_SENSORS = 7, V4_PARAMS = BASE_PARAMS + V4_SENSORS * N_NEURONS;  // 36

const std::array<std::string, MAP_H> ASCII_MAP = {
    "IIIIIIIIIIIIIIIIIIIIIII", "IWWWWWWWWWWWWWWWWWWWWWI",
    "IWPPP,PPPP,F,PPPP,PPPWI", "IWPPP,,PP,,,,,PP,,PPPWI",
    "IWPPP,,,,,,,,,,,,,PPPWI", "IWP,,WW,,,,,,,,,WW,,PWI",
    "IWHHWWW,WWWWWWW,WWWHHWI", "IWHHW,D,,,,,,,,,D,WHHWI",
    "IWHH,,W,,,WWW,,,W,,HHWI", "IW,,,,W,,,,,,,,,W,,,,WI",
    "IW,,,,WWW,,,,,WWW,,,,WI", "IW,,,,,,,,,I,,,,,,,,,WI",
    "IW,,,,WWW,,,,,WWW,,,,WI", "IW,,,,W,,,,,,,,,W,,,,WI",
    "IWHH,,W,,,WWW,,,W,,HHWI", "IWHHW,D,,,,,,,,,D,WHHWI",
    "IWHHWWW,WWWWWWW,WWWHHWI", "IWQ,,WW,,,,,,,,,WW,,QWI",
    "IWQQQ,,,,,,,,,,,,,QQQWI", "IWQQQ,,QQ,,,,,QQ,,QQQWI",
    "IWQQQ,QQQQ,G,QQQQ,QQQWI", "IWWWWWWWWWWWWWWWWWWWWWI",
    "IIIIIIIIIIIIIIIIIIIIIII",
};

struct Vec2 { double x = 0.0, y = 0.0; };
using Genome = std::array<double, N_PARAMS>;
using Sensors = std::array<double, N_SENSORS>;
using Neurons = std::array<double, N_NEURONS>;
using Lineup = std::array<int, TEAM_SIZE>;
using GenomeLineup = std::array<Genome, TEAM_SIZE>;

enum class Action { None, Noop, Forward, Backward, TurnLeft, TurnRight };
enum class Winner { Tie, Red, Blue };
enum class MatchKind { Rated, CrossTier, HallOfFame };

const char* action_name(Action a) {
    switch (a) {
        case Action::None: return "none"; case Action::Noop: return "noop";
        case Action::Forward: return "forward"; case Action::Backward: return "backward";
        case Action::TurnLeft: return "turn_left"; case Action::TurnRight: return "turn_right";
    }
    return "unknown";
}
const char* winner_name(Winner w) { return w == Winner::Red ? "red" : w == Winner::Blue ? "blue" : "tie"; }
const char* kind_name(MatchKind k) { return k == MatchKind::Rated ? "rated" : k == MatchKind::CrossTier ? "cross" : "hof"; }

class Random {
public:
    explicit Random(std::uint64_t seed) : engine_(seed) {}
    double normal(double mean = 0.0, double sd = 1.0) { return std::normal_distribution<double>(mean, sd)(engine_); }
    double uniform() { return std::uniform_real_distribution<double>(0.0, 1.0)(engine_); }
    int index(int n) {
        if (n <= 0) throw std::runtime_error("sampled empty collection");
        return std::uniform_int_distribution<int>(0, n - 1)(engine_);
    }
    template <typename T> void shuffle(T& values) { std::shuffle(values.begin(), values.end(), engine_); }
private:
    std::mt19937_64 engine_;
};

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-std::clamp(x, -30.0, 30.0))); }
double wrap_heading(double h) {
    h = std::fmod(h + PI, 2.0 * PI); if (h < 0.0) h += 2.0 * PI; return h - PI;
}
Vec2 tile_center(int row, int col) { return {(col + 0.5) * TILE, ARENA - (row + 0.5) * TILE}; }
char tile_at(double x, double y) {
    const int col = static_cast<int>(std::floor(x / TILE));
    const int row = static_cast<int>(std::floor((ARENA - y) / TILE));
    if (row < 0 || row >= MAP_H || col < 0 || col >= MAP_W) return '\0';
    return ASCII_MAP[row][col];
}
bool blocked_tile(char tile) { return tile == 'W' || tile == 'H' || tile == 'D'; }
bool blocked_position(double x, double y, double radius) {
    const std::array<Vec2, 5> samples = {{{x,y},{x+radius,y},{x-radius,y},{x,y+radius},{x,y-radius}}};
    for (const auto& p : samples) { const char t = tile_at(p.x,p.y); if (t == '\0' || blocked_tile(t)) return true; }
    return false;
}
Vec2 move_with_walls(double x, double y, double nx, double ny, double radius) {
    if (!blocked_position(nx,ny,radius)) return {nx,ny};
    if (!blocked_position(nx,y,radius)) return {nx,y};
    if (!blocked_position(x,ny,radius)) return {x,ny};
    return {x,y};
}

const std::array<Vec2, TEAM_SIZE> RED_SPAWNS = {
    tile_center(2,3), tile_center(2,8), tile_center(2,14), tile_center(2,19)};
const std::array<Vec2, TEAM_SIZE> BLUE_SPAWNS = {
    tile_center(20,3), tile_center(20,8), tile_center(20,14), tile_center(20,19)};
const Vec2 RED_HOME = tile_center(2,11), BLUE_HOME = tile_center(20,11);

std::array<double,3> relative_state(Vec2 self, double heading, Vec2 object) {
    const double dx=object.x-self.x, dy=object.y-self.y;
    return {(dx*std::cos(heading)+dy*std::sin(heading))/ARENA,
            (-dx*std::sin(heading)+dy*std::cos(heading))/ARENA,
            std::hypot(dx,dy)/MAX_DIST};
}

struct Controller {
    std::array<std::array<double,3>,3> W{};
    Neurons bias{}, tau{};
    std::array<std::array<double,3>,N_SENSORS> Ws{};
};
Controller unpack(const Genome& g) {
    Controller c; int p=0;
    for (auto& row:c.W) for(double& v:row) v=g[p++];
    for(double& v:c.bias) v=g[p++];
    Neurons raw{}; for(double& v:raw) v=g[p++];
    for(auto& row:c.Ws) for(double& v:row) v=g[p++];
    for(int i=0;i<3;++i) c.tau[i]=0.5+4.5*sigmoid(raw[i]);
    return c;
}
Action choose_action(const Neurons& out) {
    const double drive=2*out[1]-1, turn=2*out[2]-1;
    if(std::abs(drive)<0.25 && std::abs(turn)<0.25) return Action::Noop;
    if(std::abs(drive)>=std::abs(turn)) return drive>0?Action::Forward:Action::Backward;
    return turn>0?Action::TurnRight:Action::TurnLeft;
}

class Agent {
public:
    Agent(const Genome& genome, int variant, int team, Vec2 start, double h)
        : position(start), heading(h), variant_index(variant), team_index(team), controller_(unpack(genome)) {}
    void step(const Sensors& obs) {
        Neurons act{}; for(int i=0;i<3;++i) act[i]=sigmoid(state_[i]+controller_.bias[i]);
        Neurons next=state_;
        for(int n=0;n<3;++n) {
            double recurrent=0,sensory=0;
            for(int j=0;j<3;++j) recurrent+=controller_.W[n][j]*act[j];
            for(int s=0;s<N_SENSORS;++s) sensory+=controller_.Ws[s][n]*obs[s];
            next[n]+=DT*(-state_[n]+recurrent+sensory)/controller_.tau[n];
        }
        state_=next; Neurons out{}; for(int i=0;i<3;++i) out[i]=sigmoid(state_[i]+controller_.bias[i]);
        last_action=choose_action(out); apply(last_action);
    }
    Vec2 position{}; double heading=0, last_move_blocked=0; Action last_action=Action::None;
    int variant_index=-1, team_index=-1; bool carrying=false;
private:
    void apply(Action a) {
        last_move_blocked=0;
        if(a==Action::TurnLeft) heading-=DISCRETE_TURN;
        else if(a==Action::TurnRight) heading+=DISCRETE_TURN;
        else if(a==Action::Forward || a==Action::Backward) {
            const double d=a==Action::Forward?1.0:-1.0;
            const Vec2 intended={std::clamp(position.x+d*DISCRETE_STEP*std::cos(heading),AGENT_R,ARENA-AGENT_R),
                                 std::clamp(position.y+d*DISCRETE_STEP*std::sin(heading),AGENT_R,ARENA-AGENT_R)};
            const Vec2 old=position; position=move_with_walls(position.x,position.y,intended.x,intended.y,AGENT_R);
            last_move_blocked=std::hypot(position.x-old.x,position.y-old.y)<1e-6?1.0:0.0;
        }
        heading=wrap_heading(heading);
    }
    Controller controller_{}; Neurons state_{};
};

struct Flag { Vec2 home{}; int carrier=-1; };
struct AgentFrame { Vec2 pos{}; double heading=0; Action action=Action::None; bool carrying=false; double blocked=0; Sensors obs{}; };
struct Frame { int step=0; std::array<AgentFrame,AGENT_COUNT> agents{}; std::array<int,2> captures{}; };
struct MatchResult {
    Winner winner=Winner::Tie; std::array<int,2> captures{}, pickups{};
    std::array<int,AGENT_COUNT> individual_pickups{},individual_returns{};
    int steps=EP_STEPS; std::vector<Frame> trail;
};

Vec2 flag_position(const Flag& flag, const std::vector<Agent>& agents) {
    return flag.carrier>=0?agents[flag.carrier].position:flag.home;
}
Sensors make_observation(int index, const std::vector<Agent>& agents, const std::array<Flag,2>& flags) {
    const Agent& self=agents[index]; const int enemy_flag_owner=1-self.team_index;
    const Vec2 objective=self.carrying?(self.team_index==0?RED_HOME:BLUE_HOME):flag_position(flags[enemy_flag_owner],agents);
    int opponent=-1, teammate=-1; double od=1e100,td=1e100;
    for(int j=0;j<AGENT_COUNT;++j) if(j!=index) {
        const double d=std::hypot(agents[j].position.x-self.position.x,agents[j].position.y-self.position.y);
        if(agents[j].team_index==self.team_index) { if(d<td){td=d;teammate=j;} }
        else if(d<od){od=d;opponent=j;}
    }
    const auto obj=relative_state(self.position,self.heading,objective);
    const auto opp=relative_state(self.position,self.heading,agents[opponent].position);
    const auto mate=relative_state(self.position,self.heading,agents[teammate].position);
    return {obj[0],obj[1],obj[2],opp[0],opp[1],opp[2],mate[0],mate[1],mate[2],self.last_move_blocked};
}

MatchResult run_match(const GenomeLineup& red_genomes, const GenomeLineup& blue_genomes,
                      const Lineup& red_ids, const Lineup& blue_ids, std::uint64_t seed, bool record=false) {
    Random rng(seed); std::array<int,TEAM_SIZE> red_spawn={0,1,2,3},blue_spawn={0,1,2,3};
    rng.shuffle(red_spawn); rng.shuffle(blue_spawn); std::vector<Agent> agents; agents.reserve(AGENT_COUNT);
    for(int i=0;i<TEAM_SIZE;++i) agents.emplace_back(red_genomes[i],red_ids[i],0,RED_SPAWNS[red_spawn[i]],-PI/2);
    for(int i=0;i<TEAM_SIZE;++i) agents.emplace_back(blue_genomes[i],blue_ids[i],1,BLUE_SPAWNS[blue_spawn[i]],PI/2);
    std::array<Flag,2> flags={Flag{RED_HOME,-1},Flag{BLUE_HOME,-1}}; MatchResult result;
    if(record) result.trail.reserve(EP_STEPS);
    for(int t=0;t<EP_STEPS;++t) {
        std::array<Sensors,AGENT_COUNT> observations{};
        for(int i=0;i<AGENT_COUNT;++i) observations[i]=make_observation(i,agents,flags);
        for(int i=0;i<AGENT_COUNT;++i) agents[i].step(observations[i]);
        for(int i=0;i<AGENT_COUNT;++i) {
            Agent& a=agents[i]; const int enemy=1-a.team_index;
            if(!a.carrying && flags[enemy].carrier<0 && std::hypot(a.position.x-flags[enemy].home.x,a.position.y-flags[enemy].home.y)<CAPTURE_DIST) {
                flags[enemy].carrier=i; a.carrying=true; ++result.pickups[a.team_index];++result.individual_pickups[i];
            } else if(a.carrying) {
                const Vec2 home=a.team_index==0?RED_HOME:BLUE_HOME;
                if(std::hypot(a.position.x-home.x,a.position.y-home.y)<CAPTURE_DIST) {
                    ++result.captures[a.team_index]; ++result.individual_returns[i]; a.carrying=false; flags[enemy].carrier=-1;
                }
            }
        }
        if(record) {
            Frame f; f.step=t; f.captures=result.captures;
            for(int i=0;i<AGENT_COUNT;++i) f.agents[i]={agents[i].position,agents[i].heading,agents[i].last_action,agents[i].carrying,agents[i].last_move_blocked,observations[i]};
            result.trail.push_back(f);
        }
        if(result.captures[0]>=TARGET_CAPTURES || result.captures[1]>=TARGET_CAPTURES) {
            result.steps=t+1;
            result.winner=result.captures[0]>=TARGET_CAPTURES&&result.captures[1]>=TARGET_CAPTURES?Winner::Tie:
                          result.captures[0]>=TARGET_CAPTURES?Winner::Red:Winner::Blue;
            return result;
        }
    }
    result.winner=result.captures[0]>result.captures[1]?Winner::Red:result.captures[1]>result.captures[0]?Winner::Blue:Winner::Tie;
    return result;
}

// Minimal NumPy float64 I/O.
std::vector<double> load_npy(const std::string& path, std::vector<std::size_t>& shape) {
    std::ifstream in(path,std::ios::binary); if(!in) throw std::runtime_error("could not open "+path);
    char magic[6]{}; in.read(magic,6); if(std::memcmp(magic,"\x93NUMPY",6)!=0) throw std::runtime_error("not NPY: "+path);
    unsigned char version[2]{}; in.read(reinterpret_cast<char*>(version),2); std::uint32_t hlen=0;
    if(version[0]==1){std::uint16_t n=0;in.read(reinterpret_cast<char*>(&n),2);hlen=n;}else in.read(reinterpret_cast<char*>(&hlen),4);
    std::string header(hlen,'\0');in.read(header.data(),hlen);
    if(header.find("f8")==std::string::npos||header.find("True")!=std::string::npos) throw std::runtime_error("NPY must be C-order float64");
    const auto a=header.find('('),b=header.find(')',a);std::stringstream ss(header.substr(a+1,b-a-1));std::string token;
    while(std::getline(ss,token,',')){std::stringstream ts(token);std::size_t n=0;if(ts>>n)shape.push_back(n);}
    const std::size_t count=std::accumulate(shape.begin(),shape.end(),std::size_t{1},std::multiplies<std::size_t>());
    std::vector<double> data(count);in.read(reinterpret_cast<char*>(data.data()),count*sizeof(double));if(!in)throw std::runtime_error("truncated NPY");return data;
}
void save_npy(const std::string& path,const std::vector<double>& data,const std::vector<std::size_t>& shape) {
    const std::size_t expected=std::accumulate(shape.begin(),shape.end(),std::size_t{1},std::multiplies<std::size_t>());if(expected!=data.size())throw std::runtime_error("NPY shape mismatch");
    std::ostringstream st;st<<'(';for(std::size_t i=0;i<shape.size();++i){if(i)st<<", ";st<<shape[i];}if(shape.size()==1)st<<',';st<<')';
    std::string header="{'descr': '<f8', 'fortran_order': False, 'shape': "+st.str()+", }";const std::size_t pad=(16-((10+header.size()+1)%16))%16;header.append(pad,' ');header.push_back('\n');
    const std::uint16_t hlen=static_cast<std::uint16_t>(header.size());std::ofstream out(path,std::ios::binary);if(!out)throw std::runtime_error("could not write "+path);
    out.write("\x93NUMPY",6);const unsigned char v[2]={1,0};out.write(reinterpret_cast<const char*>(v),2);out.write(reinterpret_cast<const char*>(&hlen),2);out.write(header.data(),header.size());out.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(double));
}
Genome migrate_genome(const std::vector<double>& old) {
    Genome g{};
    if(old.size()==N_PARAMS){std::copy(old.begin(),old.end(),g.begin());return g;}
    if(old.size()!=V4_PARAMS)throw std::runtime_error("base genome must contain 36 or 45 values");
    std::copy(old.begin(),old.begin()+BASE_PARAMS,g.begin());
    const int old_ws=BASE_PARAMS,new_ws=BASE_PARAMS;
    for(int sensor=0;sensor<6;++sensor)for(int n=0;n<3;++n)g[new_ws+sensor*3+n]=old[old_ws+sensor*3+n];
    for(int n=0;n<3;++n)g[new_ws+9*3+n]=old[old_ws+6*3+n];
    return g;
}
Genome load_genome(const std::string& path) { std::vector<std::size_t>s;return migrate_genome(load_npy(path,s)); }
GenomeLineup load_lineup(const std::string& path) {
    std::vector<std::size_t>s;const auto data=load_npy(path,s);GenomeLineup lineup{};
    if(data.size()==N_PARAMS||data.size()==V4_PARAMS){const Genome g=migrate_genome(data);lineup.fill(g);return lineup;}
    if(data.size()!=TEAM_SIZE*N_PARAMS)throw std::runtime_error("lineup must be one genome or a 4x45 array");
    for(int i=0;i<TEAM_SIZE;++i) {
        std::copy(data.begin()+i*N_PARAMS,data.begin()+(i+1)*N_PARAMS,lineup[i].begin());
    }
    return lineup;
}
void save_genomes(const std::string& path,const std::vector<Genome>& genomes) {
    std::vector<double>d;d.reserve(genomes.size()*N_PARAMS);for(const auto&g:genomes)d.insert(d.end(),g.begin(),g.end());save_npy(path,d,{genomes.size(),N_PARAMS});
}
void save_lineup(const std::string& path,const GenomeLineup& lineup){save_genomes(path,std::vector<Genome>(lineup.begin(),lineup.end()));}

struct Variant {
    Genome genome{}; double elo=1500; int total_games=0,season_games=0,wins=0,losses=0,draws=0;
    int pickups=0,returns=0,season_pickups=0,season_returns=0;
    std::uint64_t id=0,parent=0;
};
using Pool=std::vector<Variant>;
struct ArchiveTeam { GenomeLineup genomes{}; double rating=1500; int season=-1; };

double team_rating(const Pool& pool,const Lineup& team){double r=0;for(int i:team)r+=pool[i].elo;return r/TEAM_SIZE;}
GenomeLineup team_genomes(const Pool& pool,const Lineup& team){GenomeLineup g{};for(int i=0;i<TEAM_SIZE;++i)g[i]=pool[team[i]].genome;return g;}
Lineup balanced_team(Random& rng,const Pool& pool,int focal=-1){
    std::vector<int> ids(pool.size());std::iota(ids.begin(),ids.end(),0);rng.shuffle(ids);
    std::stable_sort(ids.begin(),ids.end(),[&](int a,int b){return pool[a].season_games<pool[b].season_games;});
    Lineup t{};int p=0;
    if(focal>=0) t[p++]=focal;
    for(int id:ids){if(p>=TEAM_SIZE)break;if(id!=focal)t[p++]=id;}
    return t;
}
int least_played_focal(Random& rng,const Pool& pool){int m=std::numeric_limits<int>::max();for(const auto&v:pool)m=std::min(m,v.season_games);std::vector<int>c;for(int i=0;i<(int)pool.size();++i)if(pool[i].season_games==m)c.push_back(i);return c[rng.index(c.size())];}
Lineup matched_team(Random& rng,const Pool& pool,double target,bool cross,int focal){
    Lineup best{};double best_score=-1;
    for(int k=0;k<96;++k){Lineup candidate=balanced_team(rng,pool,focal);double diff=std::abs(team_rating(pool,candidate)-target);double score=cross?diff:-diff;if(k==0||score>best_score){best=candidate;best_score=score;}}
    return best;
}
double expected_score(double own,double other){return 1.0/(1.0+std::pow(10.0,(other-own)/400.0));}
void update_players(Pool& pool,const Lineup& team,double result,double expected){
    for(int i:team){Variant&v=pool[i];const double k=v.total_games<20?64.0:24.0;v.elo=std::clamp(v.elo+k*(result-expected),100.0,3000.0);++v.total_games;++v.season_games;if(result>0.75)++v.wins;else if(result<0.25)++v.losses;else++v.draws;}
}

struct Benchmark { int red_wins=0,blue_wins=0,draws=0;double red_caps=0,blue_caps=0; };
Benchmark benchmark(const GenomeLineup& red,const GenomeLineup& blue,int trials,int seed0){
    Benchmark b;const Lineup ids={0,1,2,3};for(int i=0;i<trials;++i){auto m=run_match(red,blue,ids,ids,seed0+i);b.red_wins+=m.winner==Winner::Red;b.blue_wins+=m.winner==Winner::Blue;b.draws+=m.winner==Winner::Tie;b.red_caps+=m.captures[0];b.blue_caps+=m.captures[1];}if(trials){b.red_caps/=trials;b.blue_caps/=trials;}return b;
}

struct Options {
    int pool=120,games=30,seasons=1000000,elite=4,checkpoint_every=100,benchmark_every=10,benchmark_trials=100;
    double mutation=0.03,max_hours=0;std::uint64_t seed=0;std::string red_base,blue_base,out="v5_elo_4v4";
};
struct LeagueState {
    Genome red_base{},blue_base{};Pool red,blue;std::vector<ArchiveTeam> red_hof,blue_hof;std::uint64_t next_id=1;
    std::vector<std::string> history;
};

Pool initial_pool(Random& rng,const Genome& base,int n,double mutation,std::uint64_t& next_id){Pool p(n);for(int i=0;i<n;++i){p[i].genome=base;if(i)for(double&v:p[i].genome)v+=rng.normal(0,mutation);p[i].id=next_id++;}return p;}
std::vector<int> ranking(const Pool& p){std::vector<int>o(p.size());std::iota(o.begin(),o.end(),0);std::stable_sort(o.begin(),o.end(),[&](int a,int b){return p[a].elo>p[b].elo;});return o;}
GenomeLineup top_lineup(const Pool& p){const auto o=ranking(p);GenomeLineup g{};for(int i=0;i<TEAM_SIZE;++i)g[i]=p[o[i]].genome;return g;}
double mean_elo(const Pool&p){double s=0;for(const auto&v:p)s+=v.elo;return s/p.size();}

void save_state(const Options&o,const LeagueState&s,int season){
    const std::string suffix=season<0?"":("_season"+[&]{std::ostringstream x;x<<std::setw(6)<<std::setfill('0')<<season;return x.str();}());
    save_genomes(o.out+suffix+"_red_base.npy",{s.red_base});save_genomes(o.out+suffix+"_blue_base.npy",{s.blue_base});
    std::vector<Genome>rg,bg;for(const auto&v:s.red)rg.push_back(v.genome);for(const auto&v:s.blue)bg.push_back(v.genome);
    save_genomes(o.out+suffix+"_red_variants.npy",rg);save_genomes(o.out+suffix+"_blue_variants.npy",bg);
    const GenomeLineup& saved_red_top=s.red_hof.back().genomes;
    const GenomeLineup& saved_blue_top=s.blue_hof.back().genomes;
    save_lineup(o.out+suffix+"_red_top4.npy",saved_red_top);save_lineup(o.out+suffix+"_blue_top4.npy",saved_blue_top);
    std::vector<Genome> red_archive,blue_archive;
    for(const auto&team:s.red_hof)red_archive.insert(red_archive.end(),team.genomes.begin(),team.genomes.end());
    for(const auto&team:s.blue_hof)blue_archive.insert(blue_archive.end(),team.genomes.begin(),team.genomes.end());
    save_genomes(o.out+suffix+"_red_hof.npy",red_archive);save_genomes(o.out+suffix+"_blue_hof.npy",blue_archive);
    std::ofstream hof_meta(o.out+suffix+"_hof.csv");hof_meta<<"color,index,season,rating\n";
    for(int i=0;i<(int)s.red_hof.size();++i)hof_meta<<"red,"<<i<<','<<s.red_hof[i].season<<','<<s.red_hof[i].rating<<'\n';
    for(int i=0;i<(int)s.blue_hof.size();++i)hof_meta<<"blue,"<<i<<','<<s.blue_hof[i].season<<','<<s.blue_hof[i].rating<<'\n';
    std::ofstream ratings(o.out+suffix+"_ratings.csv");ratings<<"color,index,id,parent,elo,total_games,wins,losses,draws\n";
    for(int c=0;c<2;++c){const Pool&p=c==0?s.red:s.blue;for(int i=0;i<(int)p.size();++i){const auto&v=p[i];ratings<<(c==0?"red":"blue")<<','<<i<<','<<v.id<<','<<v.parent<<','<<v.elo<<','<<v.total_games<<','<<v.wins<<','<<v.losses<<','<<v.draws<<'\n';}}
    std::ofstream history(o.out+"_history.csv");history<<"season,red_best_elo,blue_best_elo,red_mean_elo,blue_mean_elo,matches,rated_matches,cross_matches,hof_matches,red_wins,blue_wins,draws,red_bench_win,blue_bench_win,bench_draw,red_mean_caps,blue_mean_caps,red_vs_anchor_win,blue_vs_anchor_win\n";for(const auto&line:s.history)history<<line<<'\n';
}

void reproduce(Random&rng,Pool&pool,Genome&base,int elite,double mutation,std::uint64_t&next_id){
    const auto order=ranking(pool);base=pool[order[0]].genome;Pool next;next.reserve(pool.size());
    for(int i=0;i<std::min<int>(elite,pool.size());++i){Variant v=pool[order[i]];v.season_games=0;next.push_back(v);}
    while(next.size()<pool.size()){Variant child;child.genome=base;for(double&v:child.genome)v+=rng.normal(0,mutation);child.id=next_id++;child.parent=pool[order[0]].id;next.push_back(child);}pool=std::move(next);
}

void run_league(const Options&o){
    Random rng(o.seed);LeagueState s;s.red_base=load_genome(o.red_base);s.blue_base=load_genome(o.blue_base);
    s.red=initial_pool(rng,s.red_base,o.pool,o.mutation,s.next_id);s.blue=initial_pool(rng,s.blue_base,o.pool,o.mutation,s.next_id);
    s.red_hof.push_back({GenomeLineup{s.red_base,s.red_base,s.red_base,s.red_base},1500,-1});
    s.blue_hof.push_back({GenomeLineup{s.blue_base,s.blue_base,s.blue_base,s.blue_base},1500,-1});
    const auto started=std::chrono::steady_clock::now();
    for(int season=0;season<o.seasons;++season){
        for(auto&v:s.red)v.season_games=0;
        for(auto&v:s.blue)v.season_games=0;
        int matches=0,rated_matches=0,cross_matches=0,hof_matches=0,rw=0,bw=0,dr=0;
        auto unfinished=[&](const Pool&p){return std::any_of(p.begin(),p.end(),[&](const Variant&v){return v.season_games<o.games;});};
        while(unfinished(s.red)||unfinished(s.blue)){
            const bool red_need=unfinished(s.red),blue_need=unfinished(s.blue);double roll=rng.uniform();MatchKind kind=roll<0.8?MatchKind::Rated:roll<0.9?MatchKind::CrossTier:MatchKind::HallOfFame;
            if(!red_need||!blue_need)kind=MatchKind::HallOfFame;
            rated_matches += kind==MatchKind::Rated;
            cross_matches += kind==MatchKind::CrossTier;
            hof_matches += kind==MatchKind::HallOfFame;
            Lineup rt{},bt{};GenomeLineup rg{},bg{};bool red_live=true,blue_live=true;double rr=1500,br=1500;
            if(kind==MatchKind::HallOfFame){
                bool challenge_red=red_need&&(!blue_need||rng.uniform()<0.5);
                if(challenge_red){rt=balanced_team(rng,s.red,least_played_focal(rng,s.red));rg=team_genomes(s.red,rt);rr=team_rating(s.red,rt);const auto&h=s.blue_hof[rng.index(s.blue_hof.size())];bg=h.genomes;br=h.rating;blue_live=false;bt={-1,-1,-1,-1};}
                else {bt=balanced_team(rng,s.blue,least_played_focal(rng,s.blue));bg=team_genomes(s.blue,bt);br=team_rating(s.blue,bt);const auto&h=s.red_hof[rng.index(s.red_hof.size())];rg=h.genomes;rr=h.rating;red_live=false;rt={-1,-1,-1,-1};}
            }else{
                rt=balanced_team(rng,s.red,least_played_focal(rng,s.red));rr=team_rating(s.red,rt);bt=matched_team(rng,s.blue,rr,kind==MatchKind::CrossTier,least_played_focal(rng,s.blue));br=team_rating(s.blue,bt);rg=team_genomes(s.red,rt);bg=team_genomes(s.blue,bt);
            }
            const auto m=run_match(rg,bg,rt,bt,o.seed+season*1000003ULL+matches*37ULL);const double rs=m.winner==Winner::Red?1:m.winner==Winner::Tie?0.5:0,bs=1-rs;const double re=expected_score(rr,br);
            if(red_live) update_players(s.red,rt,rs,re);
            if(blue_live) update_players(s.blue,bt,bs,1-re);
            ++matches;rw+=m.winner==Winner::Red;bw+=m.winner==Winner::Blue;dr+=m.winner==Winner::Tie;
        }
        const auto ro=ranking(s.red),bo=ranking(s.blue);Benchmark bench{},red_anchor{},blue_anchor{};bool did_bench=o.benchmark_every>0&&(season%o.benchmark_every==0||season==o.seasons-1);
        if(did_bench){bench=benchmark(top_lineup(s.red),top_lineup(s.blue),o.benchmark_trials,900000);red_anchor=benchmark(top_lineup(s.red),s.blue_hof.front().genomes,o.benchmark_trials,910000);blue_anchor=benchmark(s.red_hof.front().genomes,top_lineup(s.blue),o.benchmark_trials,920000);}
        std::ostringstream line;line<<season<<','<<s.red[ro[0]].elo<<','<<s.blue[bo[0]].elo<<','<<mean_elo(s.red)<<','<<mean_elo(s.blue)<<','<<matches<<','<<rated_matches<<','<<cross_matches<<','<<hof_matches<<','<<rw<<','<<bw<<','<<dr<<',';
        if(did_bench)line<<100.0*bench.red_wins/o.benchmark_trials<<','<<100.0*bench.blue_wins/o.benchmark_trials<<','<<100.0*bench.draws/o.benchmark_trials<<','<<bench.red_caps<<','<<bench.blue_caps<<','<<100.0*red_anchor.red_wins/o.benchmark_trials<<','<<100.0*blue_anchor.blue_wins/o.benchmark_trials;else line<<",,,,,,";s.history.push_back(line.str());
        std::cout<<"season "<<std::setw(5)<<season<<std::fixed<<std::setprecision(1)<<" red="<<s.red[ro[0]].elo<<" blue="<<s.blue[bo[0]].elo<<" mean=("<<mean_elo(s.red)<<','<<mean_elo(s.blue)<<") matches="<<matches<<" W/L/D="<<rw<<'/'<<bw<<'/'<<dr;
        std::cout<<" mix="<<rated_matches<<'/'<<cross_matches<<'/'<<hof_matches;
        if(did_bench) std::cout<<" bench="<<bench.red_wins<<'/'<<bench.blue_wins<<'/'<<bench.draws<<" caps="<<std::setprecision(2)<<bench.red_caps<<'/'<<bench.blue_caps<<" anchors="<<red_anchor.red_wins<<'/'<<blue_anchor.blue_wins;
        std::cout<<'\n';
        GenomeLineup red_top=top_lineup(s.red),blue_top=top_lineup(s.blue);double red_rating=0,blue_rating=0;for(int i=0;i<TEAM_SIZE;++i){red_rating+=s.red[ro[i]].elo;blue_rating+=s.blue[bo[i]].elo;}red_rating/=TEAM_SIZE;blue_rating/=TEAM_SIZE;
        s.red_hof.push_back({red_top,red_rating,season});s.blue_hof.push_back({blue_top,blue_rating,season});if(s.red_hof.size()>20)s.red_hof.erase(s.red_hof.begin()+1);if(s.blue_hof.size()>20)s.blue_hof.erase(s.blue_hof.begin()+1);
        reproduce(rng,s.red,s.red_base,o.elite,o.mutation,s.next_id);reproduce(rng,s.blue,s.blue_base,o.elite,o.mutation,s.next_id);
        if(o.checkpoint_every>0&&((season+1)%o.checkpoint_every==0||season==o.seasons-1))save_state(o,s,season+1);
        if(o.max_hours>0){const std::chrono::duration<double,std::ratio<3600>>elapsed=std::chrono::steady_clock::now()-started;if(elapsed.count()>=o.max_hours){save_state(o,s,season+1);std::cout<<"wall-clock limit reached at season "<<season+1<<" after "<<std::setprecision(4)<<elapsed.count()<<" hours\n";break;}}
    }
    save_state(o,s,-1);
}

void save_replay(const std::string&path,const MatchResult&m){std::ofstream out(path);if(!out)throw std::runtime_error("could not write replay");out<<"step,red_captures,blue_captures";for(int i=0;i<AGENT_COUNT;++i)out<<",a"<<i<<"_x,a"<<i<<"_y,a"<<i<<"_heading,a"<<i<<"_action,a"<<i<<"_carrying,a"<<i<<"_blocked";out<<'\n';out<<std::setprecision(17);for(const auto&f:m.trail){out<<f.step<<','<<f.captures[0]<<','<<f.captures[1];for(const auto&a:f.agents)out<<','<<a.pos.x<<','<<a.pos.y<<','<<a.heading<<','<<action_name(a.action)<<','<<a.carrying<<','<<a.blocked;out<<'\n';}}

void help(){std::cout<<"V5 Elo 4v4 CTF\nTraining: v5_elo_4v4.exe --red-base FILE --blue-base FILE [--pool 120 --games 30 --seasons N --mut .03 --elite 4 --seed N --checkpoint-every N --benchmark-every N --benchmark-trials N --max-hours H --out PREFIX]\nReplay: v5_elo_4v4.exe --replay RED_TOP4.npy BLUE_TOP4.npy [--replay-seed N --replay-out FILE]\n";}
struct Command {Options o;bool replay=false,quick=false;std::string rr,rb,replay_out="v5_replay.csv";std::uint64_t replay_seed=42;};
Command parse(int argc,char**argv){Command c;auto value=[&](int&i,const std::string&x){if(++i>=argc)throw std::runtime_error(x+" requires value");return std::string(argv[i]);};for(int i=1;i<argc;++i){std::string x=argv[i];if(x=="--help"||x=="-h"){help();std::exit(0);}else if(x=="--quick")c.quick=true;else if(x=="--pool")c.o.pool=std::stoi(value(i,x));else if(x=="--games")c.o.games=std::stoi(value(i,x));else if(x=="--seasons")c.o.seasons=std::stoi(value(i,x));else if(x=="--mut")c.o.mutation=std::stod(value(i,x));else if(x=="--elite")c.o.elite=std::stoi(value(i,x));else if(x=="--seed")c.o.seed=std::stoull(value(i,x));else if(x=="--red-base")c.o.red_base=value(i,x);else if(x=="--blue-base")c.o.blue_base=value(i,x);else if(x=="--checkpoint-every")c.o.checkpoint_every=std::stoi(value(i,x));else if(x=="--benchmark-every")c.o.benchmark_every=std::stoi(value(i,x));else if(x=="--benchmark-trials")c.o.benchmark_trials=std::stoi(value(i,x));else if(x=="--max-hours")c.o.max_hours=std::stod(value(i,x));else if(x=="--out")c.o.out=value(i,x);else if(x=="--replay"){c.replay=true;c.rr=value(i,x);c.rb=value(i,x);}else if(x=="--replay-seed")c.replay_seed=std::stoull(value(i,x));else if(x=="--replay-out")c.replay_out=value(i,x);else throw std::runtime_error("unknown option "+x);}if(c.quick){c.o.pool=16;c.o.games=4;c.o.seasons=2;c.o.benchmark_every=1;c.o.benchmark_trials=10;}if(!c.replay&&(c.o.red_base.empty()||c.o.blue_base.empty()))throw std::runtime_error("--red-base and --blue-base are required");if(c.o.pool<TEAM_SIZE||c.o.elite<0||c.o.elite>c.o.pool||c.o.games<1)throw std::runtime_error("invalid pool, elite, or games");return c;}

} // namespace league

#ifndef LEAGUE_NO_MAIN
int main(int argc,char**argv){try{const auto c=league::parse(argc,argv);if(c.replay){const auto r=league::load_lineup(c.rr),b=league::load_lineup(c.rb);const league::Lineup ids={0,1,2,3};const auto m=league::run_match(r,b,ids,ids,c.replay_seed,true);league::save_replay(c.replay_out,m);std::cout<<"winner="<<league::winner_name(m.winner)<<" captures=["<<m.captures[0]<<','<<m.captures[1]<<"] steps="<<m.steps<<" saved="<<c.replay_out<<'\n';}else league::run_league(c.o);return 0;}catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<'\n';return 1;}}
#endif
