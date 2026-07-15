#define SCRIMMAGE_NO_MAIN
#include "v5_red_scrimmage.cpp"

namespace curriculum {

using namespace league;
using scrimmage::Archive;
using scrimmage::State;

struct Options {
    int games=30,seasons=1000000,elite=2,checkpoint_every=1000,benchmark_every=25,benchmark_trials=100,mastery_streak=3;
    double mutation=0.03,max_hours=8.0;std::uint64_t seed=0;
    std::string seed_team,seed_base,curriculum,out="v5_red_blue_curriculum";
};

struct CrossBench {int red_wins=0,blue_wins=0,draws=0;double red_caps=0,blue_caps=0;};
struct CurriculumState {State red;std::vector<GenomeLineup> blue_stages;int stage=0,mastery=0;};

std::vector<GenomeLineup> load_stages(const std::string&path){
    std::vector<std::size_t>shape;const auto data=load_npy(path,shape);
    if(data.size()%(TEAM_SIZE*N_PARAMS)!=0)throw std::runtime_error("curriculum must contain Nx4x45 genomes");
    const int count=static_cast<int>(data.size()/(TEAM_SIZE*N_PARAMS));std::vector<GenomeLineup>stages(count);
    for(int s=0;s<count;++s)for(int i=0;i<TEAM_SIZE;++i)std::copy(data.begin()+(s*TEAM_SIZE+i)*N_PARAMS,data.begin()+(s*TEAM_SIZE+i+1)*N_PARAMS,stages[s][i].begin());
    return stages;
}

CrossBench cross_benchmark(const GenomeLineup&red,const GenomeLineup&blue,int trials,int seed0){
    CrossBench b;const Lineup ids={0,1,2,3};
    for(int i=0;i<trials;++i){const auto m=run_match(red,blue,ids,ids,seed0+i);b.red_wins+=m.winner==Winner::Red;b.blue_wins+=m.winner==Winner::Blue;b.draws+=m.winner==Winner::Tie;b.red_caps+=m.captures[0];b.blue_caps+=m.captures[1];}
    if(trials){b.red_caps/=trials;b.blue_caps/=trials;}return b;
}

void update_cross(Pool&pool,const Lineup&team,const MatchResult&m,double opponent_rating){
    const double result=m.winner==Winner::Red?1.0:m.winner==Winner::Tie?0.5:0.0;
    const double expected=expected_score(team_rating(pool,team),opponent_rating);
    for(int slot=0;slot<TEAM_SIZE;++slot){Variant&v=pool[team[slot]];const double k=v.total_games<20?64.0:24.0;v.elo=std::clamp(v.elo+k*(result-expected),100.0,3000.0);++v.total_games;++v.season_games;if(result>0.75)++v.wins;else if(result<0.25)++v.losses;else++v.draws;v.pickups+=m.individual_pickups[slot];v.returns+=m.individual_returns[slot];v.season_pickups+=m.individual_pickups[slot];v.season_returns+=m.individual_returns[slot];}
}

bool no_worse(const CrossBench&a,const CrossBench&b){
    const double an=a.red_wins+0.5*a.draws,bn=b.red_wins+0.5*b.draws;
    return a.red_caps>b.red_caps+1e-12||(std::abs(a.red_caps-b.red_caps)<1e-12&&an>=bn);
}

void save(const Options&o,const CurriculumState&c,int season){
    std::string suffix;if(season>=0){std::ostringstream x;x<<"_season"<<std::setw(6)<<std::setfill('0')<<season;suffix=x.str();}
    save_genomes(o.out+suffix+"_base.npy",{c.red.base});std::vector<Genome>g;for(const auto&v:c.red.variants)g.push_back(v.genome);save_genomes(o.out+suffix+"_variants.npy",g);
    save_lineup(o.out+suffix+"_objective_champion_team.npy",c.red.hof.back().genomes);save_lineup(o.out+suffix+"_elo_champion_team.npy",c.red.last_elo_champion);
    std::vector<Genome>archive;for(const auto&h:c.red.hof)archive.insert(archive.end(),h.genomes.begin(),h.genomes.end());save_genomes(o.out+suffix+"_red_hof.npy",archive);
    std::ofstream ratings(o.out+suffix+"_ratings.csv");ratings<<"index,id,parent,elo,total_games,wins,losses,draws,pickups,returns\n";for(int i=0;i<(int)c.red.variants.size();++i){const auto&v=c.red.variants[i];ratings<<i<<','<<v.id<<','<<v.parent<<','<<v.elo<<','<<v.total_games<<','<<v.wins<<','<<v.losses<<','<<v.draws<<','<<v.pickups<<','<<v.returns<<'\n';}
    std::ofstream history(o.out+"_history.csv");history<<"season,trained_stage,next_stage,mastery,best_objective,best_returns,best_pickups,best_elo,mean_elo,matches,self_matches,current_blue_matches,next_blue_matches,promoted,anchor_candidate_caps,anchor_incumbent_caps,stage_candidate_score,stage_incumbent_score,current_red_wins,current_blue_wins,current_draws,current_red_caps,current_blue_caps,final_red_wins,final_blue_wins,final_draws,final_red_caps,final_blue_caps\n";for(const auto&line:c.red.history)history<<line<<'\n';
    std::ofstream stage_file(o.out+suffix+"_stage.txt");stage_file<<c.stage<<'\n'<<c.mastery<<'\n';
}

void run(const Options&o){
    Random rng(o.seed);CurriculumState c;c.blue_stages=load_stages(o.curriculum);const GenomeLineup seed=load_lineup(o.seed_team);c.red.base=load_genome(o.seed_base);c.red.last_elo_champion=seed;c.red.variants.resize(8);
    for(int i=0;i<8;++i){c.red.variants[i].genome=seed[i%4];if(i>=4)for(double&v:c.red.variants[i].genome)v+=rng.normal(0,o.mutation);c.red.variants[i].id=c.red.next_id++;}
    c.red.hof.push_back({seed,1500,-1});const auto started=std::chrono::steady_clock::now();
    for(int season=0;season<o.seasons;++season){
        const int trained_stage=c.stage;
        for(auto&v:c.red.variants){v.season_games=0;v.season_pickups=0;v.season_returns=0;}
        int matches=0,self_matches=0,current_matches=0,next_matches=0;
        auto unfinished=[&]{return std::any_of(c.red.variants.begin(),c.red.variants.end(),[&](const Variant&v){return v.season_games<o.games;});};
        while(unfinished()){
            const double roll=rng.uniform();
            if(roll<0.60){
                ++self_matches;const auto p=scrimmage::choose_partition(rng,c.red.variants,false);const double ar=team_rating(c.red.variants,p.first),br=team_rating(c.red.variants,p.second);const auto series=scrimmage::paired_series(team_genomes(c.red.variants,p.first),team_genomes(c.red.variants,p.second),p.first,p.second,o.seed+season*1000003ULL+matches*2ULL);const double expected=expected_score(ar,br),leg1=scrimmage::score_for_side(series.legs[0].winner,0),leg2=scrimmage::score_for_side(series.legs[1].winner,1);scrimmage::update_team(c.red.variants,p.first,series.first_score,expected,leg1,leg2);scrimmage::update_team(c.red.variants,p.second,1-series.first_score,1-expected,1-leg1,1-leg2);scrimmage::add_contributions(c.red.variants,p.first,series,true);scrimmage::add_contributions(c.red.variants,p.second,series,false);
            }else{
                const bool challenge=roll>=0.90&&c.stage+1<(int)c.blue_stages.size();const int stage=challenge?c.stage+1:c.stage;if(challenge)++next_matches;else ++current_matches;const Lineup live=scrimmage::least_played_team(rng,c.red.variants);const auto m=run_match(team_genomes(c.red.variants,live),c.blue_stages[stage],live,Lineup{0,1,2,3},o.seed+season*1000003ULL+matches*2ULL);update_cross(c.red.variants,live,m,1500+150*stage);
            }
            ++matches;
        }
        const auto objective=scrimmage::objective_rank(c.red.variants),elo=scrimmage::elo_rank(c.red.variants);const GenomeLineup champion=scrimmage::lineup_from_order(c.red.variants,objective);c.red.last_elo_champion=scrimmage::lineup_from_order(c.red.variants,elo);
        GenomeLineup candidate{},incumbent{};candidate.fill(c.red.variants[objective[0]].genome);incumbent.fill(c.red.base);
        const auto candidate_anchor=scrimmage::benchmark_team(candidate,c.red.hof.front().genomes,30,930000);const auto incumbent_anchor=scrimmage::benchmark_team(incumbent,c.red.hof.front().genomes,30,930000);
        const auto candidate_stage=cross_benchmark(candidate,c.blue_stages[c.stage],30,940000);const auto incumbent_stage=cross_benchmark(incumbent,c.blue_stages[c.stage],30,940000);
        const bool anchor_ok=candidate_anchor.champion_caps+1e-12>=incumbent_anchor.champion_caps;const bool stage_ok=no_worse(candidate_stage,incumbent_stage);const bool promoted=anchor_ok&&stage_ok;if(promoted)c.red.base=c.red.variants[objective[0]].genome;
        CrossBench current{},final{};const bool did_bench=o.benchmark_every>0&&(season%o.benchmark_every==0||season==o.seasons-1);
        if(did_bench){current=cross_benchmark(champion,c.blue_stages[c.stage],o.benchmark_trials,900000);final=cross_benchmark(champion,c.blue_stages.back(),o.benchmark_trials,910000);const bool mastered=current.red_caps>=0.95&&(current.red_wins+current.draws)>=0.60*o.benchmark_trials;if(mastered)++c.mastery;else c.mastery=0;if(c.mastery>=o.mastery_streak&&c.stage+1<(int)c.blue_stages.size()){++c.stage;c.mastery=0;}}
        const int objective_score=100*c.red.variants[objective[0]].season_returns+25*c.red.variants[objective[0]].season_pickups;double rating=0;for(int i=0;i<4;++i)rating+=c.red.variants[objective[i]].elo;rating/=4;
        std::ostringstream line;line<<season<<','<<trained_stage<<','<<c.stage<<','<<c.mastery<<','<<objective_score<<','<<c.red.variants[objective[0]].season_returns<<','<<c.red.variants[objective[0]].season_pickups<<','<<c.red.variants[elo[0]].elo<<','<<scrimmage::mean_rating(c.red.variants)<<','<<matches<<','<<self_matches<<','<<current_matches<<','<<next_matches<<','<<promoted<<','<<candidate_anchor.champion_caps<<','<<incumbent_anchor.champion_caps<<','<<(candidate_stage.red_wins+0.5*candidate_stage.draws)<<','<<(incumbent_stage.red_wins+0.5*incumbent_stage.draws)<<',';
        if(did_bench)line<<current.red_wins<<','<<current.blue_wins<<','<<current.draws<<','<<current.red_caps<<','<<current.blue_caps<<','<<final.red_wins<<','<<final.blue_wins<<','<<final.draws<<','<<final.red_caps<<','<<final.blue_caps;else line<<",,,,,,,,,,";c.red.history.push_back(line.str());
        std::cout<<"season "<<std::setw(6)<<season<<" stage="<<trained_stage<<"->"<<c.stage<<'/'<<c.blue_stages.size()-1<<" mastery="<<c.mastery<<" objective="<<objective_score<<" events="<<c.red.variants[objective[0]].season_returns<<'/'<<c.red.variants[objective[0]].season_pickups<<std::fixed<<std::setprecision(1)<<" elo="<<c.red.variants[elo[0]].elo<<" promoted="<<promoted<<" mix="<<self_matches<<'/'<<current_matches<<'/'<<next_matches;
        if(did_bench)std::cout<<" current="<<current.red_wins<<'/'<<current.blue_wins<<'/'<<current.draws<<" caps="<<std::setprecision(2)<<current.red_caps<<'/'<<current.blue_caps<<" final="<<final.red_wins<<'/'<<final.blue_wins<<'/'<<final.draws;
        std::cout<<'\n';
        c.red.hof.push_back({champion,rating,season});if(c.red.hof.size()>20)c.red.hof.erase(c.red.hof.begin()+1);scrimmage::reproduce(rng,c.red,scrimmage::Options{o.games,o.seasons,o.elite,o.checkpoint_every,o.benchmark_every,o.benchmark_trials,o.mutation,o.max_hours,o.seed,o.seed_team,o.out});
        if(o.checkpoint_every>0&&((season+1)%o.checkpoint_every==0||season==o.seasons-1))save(o,c,season+1);
        if(o.max_hours>0){const std::chrono::duration<double,std::ratio<3600>>elapsed=std::chrono::steady_clock::now()-started;if(elapsed.count()>=o.max_hours){save(o,c,season+1);std::cout<<"wall-clock limit reached at season "<<season+1<<" after "<<std::setprecision(4)<<elapsed.count()<<" hours\n";break;}}
    }
    save(o,c,-1);
}

Options parse(int argc,char**argv){Options o;auto value=[&](int&i,const std::string&x){if(++i>=argc)throw std::runtime_error(x+" requires value");return std::string(argv[i]);};for(int i=1;i<argc;++i){std::string x=argv[i];if(x=="--seed-team")o.seed_team=value(i,x);else if(x=="--seed-base")o.seed_base=value(i,x);else if(x=="--curriculum")o.curriculum=value(i,x);else if(x=="--games")o.games=std::stoi(value(i,x));else if(x=="--seasons")o.seasons=std::stoi(value(i,x));else if(x=="--elite")o.elite=std::stoi(value(i,x));else if(x=="--mut")o.mutation=std::stod(value(i,x));else if(x=="--seed")o.seed=std::stoull(value(i,x));else if(x=="--checkpoint-every")o.checkpoint_every=std::stoi(value(i,x));else if(x=="--benchmark-every")o.benchmark_every=std::stoi(value(i,x));else if(x=="--benchmark-trials")o.benchmark_trials=std::stoi(value(i,x));else if(x=="--mastery-streak")o.mastery_streak=std::stoi(value(i,x));else if(x=="--max-hours")o.max_hours=std::stod(value(i,x));else if(x=="--out")o.out=value(i,x);else if(x=="--quick"){o.games=4;o.seasons=4;o.benchmark_every=1;o.benchmark_trials=20;o.mastery_streak=2;}else throw std::runtime_error("unknown option "+x);}if(o.seed_team.empty()||o.seed_base.empty()||o.curriculum.empty())throw std::runtime_error("seed-team, seed-base, and curriculum are required");return o;}

} // namespace curriculum

int main(int argc,char**argv){try{curriculum::run(curriculum::parse(argc,argv));return 0;}catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<'\n';return 1;}}
