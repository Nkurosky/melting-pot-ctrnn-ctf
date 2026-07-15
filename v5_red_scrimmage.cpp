#define LEAGUE_NO_MAIN
#include "v5_elo_4v4.cpp"

namespace scrimmage {

using namespace league;

struct Options {
    int games=30,seasons=1000000,elite=2,checkpoint_every=100,benchmark_every=10,benchmark_trials=100;
    double mutation=0.03,max_hours=0;std::uint64_t seed=0;
    std::string seed_team,out="v5_red_scrimmage";
};

struct Partition { Lineup first{},second{}; };
struct Archive { GenomeLineup genomes{};double rating=1500;int season=-1; };
struct Bench {int champion_wins=0,anchor_wins=0,draws=0;double champion_caps=0,anchor_caps=0;};
struct State {
    Pool variants;Genome base{};std::vector<Archive> hof;std::uint64_t next_id=1;
    GenomeLineup last_elo_champion{};
    std::vector<std::string> history;
};

double score_for_side(Winner winner,int side){
    if(winner==Winner::Tie)return 0.5;
    return (winner==Winner::Red&&side==0)||(winner==Winner::Blue&&side==1)?1.0:0.0;
}

Partition candidate_partition(Random&rng){
    std::array<int,8> ids={0,1,2,3,4,5,6,7};rng.shuffle(ids);Partition p;
    for(int i=0;i<4;++i){p.first[i]=ids[i];p.second[i]=ids[i+4];}return p;
}

Partition choose_partition(Random&rng,const Pool&pool,bool cross){
    Partition best{};double best_score=-1;
    for(int attempt=0;attempt<128;++attempt){
        Partition p=candidate_partition(rng);
        double diff=std::abs(team_rating(pool,p.first)-team_rating(pool,p.second));
        double imbalance=0;for(int i:p.first)imbalance+=pool[i].season_games;for(int i:p.second)imbalance+=pool[i].season_games;
        (void)imbalance;
        const double score=cross?diff:-diff;
        if(attempt==0||score>best_score){best=p;best_score=score;}
    }
    return best;
}

Lineup least_played_team(Random&rng,const Pool&pool){
    std::vector<int>ids(pool.size());std::iota(ids.begin(),ids.end(),0);rng.shuffle(ids);
    std::stable_sort(ids.begin(),ids.end(),[&](int a,int b){return pool[a].season_games<pool[b].season_games;});
    Lineup team{};for(int i=0;i<4;++i)team[i]=ids[i];return team;
}

void update_team(Pool&pool,const Lineup&team,double result,double expected,
                 double first_leg,double second_leg){
    for(int index:team){
        Variant&v=pool[index];const double k=v.total_games<20?64.0:24.0;
        v.elo=std::clamp(v.elo+k*(result-expected),100.0,3000.0);
        v.total_games+=2;v.season_games+=2;
        for(double leg:{first_leg,second_leg}){if(leg>0.75)++v.wins;else if(leg<0.25)++v.losses;else++v.draws;}
    }
}

struct SeriesResult {double first_score=0.5;std::array<MatchResult,2> legs{};};
SeriesResult paired_series(const GenomeLineup&first,const GenomeLineup&second,
                           const Lineup&first_ids,const Lineup&second_ids,std::uint64_t seed){
    SeriesResult s;
    s.legs[0]=run_match(first,second,first_ids,second_ids,seed);
    s.legs[1]=run_match(second,first,second_ids,first_ids,seed+1);
    const double leg1=score_for_side(s.legs[0].winner,0);
    const double leg2=score_for_side(s.legs[1].winner,1);
    s.first_score=0.5*(leg1+leg2);
    return s;
}

Bench benchmark_team(const GenomeLineup&champion,const GenomeLineup&anchor,int trials,int seed0){
    Bench b;const Lineup ids={0,1,2,3};
    for(int i=0;i<trials;++i){
        const auto s=paired_series(champion,anchor,ids,ids,seed0+2*i);
        if(s.first_score>0.5)++b.champion_wins;else if(s.first_score<0.5)++b.anchor_wins;else ++b.draws;
        b.champion_caps+=0.5*(s.legs[0].captures[0]+s.legs[1].captures[1]);
        b.anchor_caps+=0.5*(s.legs[0].captures[1]+s.legs[1].captures[0]);
    }
    if(trials){b.champion_caps/=trials;b.anchor_caps/=trials;}return b;
}

void add_contributions(Pool&pool,const Lineup&team,const SeriesResult&series,bool first_team){
    for(int slot=0;slot<TEAM_SIZE;++slot){
        const int first_index=first_team?slot:slot+TEAM_SIZE;
        const int second_index=first_team?slot+TEAM_SIZE:slot;
        const int pickups=series.legs[0].individual_pickups[first_index]+series.legs[1].individual_pickups[second_index];
        const int returns=series.legs[0].individual_returns[first_index]+series.legs[1].individual_returns[second_index];
        Variant&v=pool[team[slot]];v.pickups+=pickups;v.returns+=returns;v.season_pickups+=pickups;v.season_returns+=returns;
    }
}

std::vector<int> elo_rank(const Pool&pool){
    std::vector<int>order(pool.size());std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](int a,int b){return pool[a].elo>pool[b].elo;});return order;
}

std::vector<int> objective_rank(const Pool&pool){
    std::vector<int>order(pool.size());std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](int a,int b){
        if(pool[a].season_returns!=pool[b].season_returns)return pool[a].season_returns>pool[b].season_returns;
        if(pool[a].season_pickups!=pool[b].season_pickups)return pool[a].season_pickups>pool[b].season_pickups;
        return pool[a].elo>pool[b].elo;
    });return order;
}

GenomeLineup lineup_from_order(const Pool&pool,const std::vector<int>&order){GenomeLineup team{};for(int i=0;i<4;++i)team[i]=pool[order[i]].genome;return team;}
GenomeLineup objective_top_four(const Pool&pool){return lineup_from_order(pool,objective_rank(pool));}
GenomeLineup elo_top_four(const Pool&pool){return lineup_from_order(pool,elo_rank(pool));}
double mean_rating(const Pool&pool){double total=0;for(const auto&v:pool)total+=v.elo;return total/pool.size();}

void save(const Options&o,const State&s,int season){
    std::string suffix;
    if(season>=0){std::ostringstream x;x<<"_season"<<std::setw(6)<<std::setfill('0')<<season;suffix=x.str();}
    save_genomes(o.out+suffix+"_base.npy",{s.base});
    std::vector<Genome>genomes;for(const auto&v:s.variants)genomes.push_back(v.genome);
    save_genomes(o.out+suffix+"_variants.npy",genomes);
    save_lineup(o.out+suffix+"_objective_champion_team.npy",s.hof.back().genomes);
    save_lineup(o.out+suffix+"_elo_champion_team.npy",s.last_elo_champion);
    std::vector<Genome>archive;for(const auto&h:s.hof)archive.insert(archive.end(),h.genomes.begin(),h.genomes.end());
    save_genomes(o.out+suffix+"_hof.npy",archive);
    std::ofstream ratings(o.out+suffix+"_ratings.csv");ratings<<"index,id,parent,elo,total_games,wins,losses,draws,pickups,returns\n";
    for(int i=0;i<(int)s.variants.size();++i){const auto&v=s.variants[i];ratings<<i<<','<<v.id<<','<<v.parent<<','<<v.elo<<','<<v.total_games<<','<<v.wins<<','<<v.losses<<','<<v.draws<<','<<v.pickups<<','<<v.returns<<'\n';}
    std::ofstream history(o.out+"_history.csv");history<<"season,best_objective,best_returns,best_pickups,best_elo,mean_elo,series,balanced,cross,hof,top_side_wins,bottom_side_wins,leg_draws,promoted,candidate_gate_wins,incumbent_gate_wins,candidate_gate_caps,incumbent_gate_caps,champion_bench_wins,anchor_bench_wins,bench_draws,champion_mean_caps,anchor_mean_caps\n";for(const auto&line:s.history)history<<line<<'\n';
}

void reproduce(Random&rng,State&s,const Options&o){
    const auto order=objective_rank(s.variants);Pool next;next.reserve(8);
    for(int i=0;i<o.elite;++i){Variant v=s.variants[order[i]];v.season_games=0;next.push_back(v);}
    while(next.size()<8){Variant child;child.genome=s.base;for(double&v:child.genome)v+=rng.normal(0,o.mutation);child.id=s.next_id++;child.parent=s.variants[order[0]].id;next.push_back(child);}s.variants=std::move(next);
}

void run(const Options&o){
    Random rng(o.seed);State s;const GenomeLineup seed=load_lineup(o.seed_team);s.base=seed[0];s.variants.resize(8);
    for(int i=0;i<8;++i){s.variants[i].genome=seed[i%4];if(i>=4)for(double&v:s.variants[i].genome)v+=rng.normal(0,o.mutation);s.variants[i].id=s.next_id++;}
    s.last_elo_champion=seed;s.hof.push_back({seed,1500,-1});const auto started=std::chrono::steady_clock::now();
    for(int season=0;season<o.seasons;++season){
        for(auto&v:s.variants){v.season_games=0;v.season_pickups=0;v.season_returns=0;}
        int series_count=0,balanced=0,cross=0,hof_count=0,top_wins=0,bottom_wins=0,leg_draws=0;
        auto unfinished=[&]{return std::any_of(s.variants.begin(),s.variants.end(),[&](const Variant&v){return v.season_games<o.games;});};
        while(unfinished()){
            const double roll=rng.uniform();const bool hof_match=roll>=0.9;const bool cross_match=!hof_match&&roll>=0.8;
            if(hof_match){
                ++hof_count;const Lineup live=least_played_team(rng,s.variants);const auto&archive=s.hof[rng.index(s.hof.size())];const double live_rating=team_rating(s.variants,live);const GenomeLineup live_g=team_genomes(s.variants,live);const auto paired=paired_series(live_g,archive.genomes,live,Lineup{0,1,2,3},o.seed+season*1000003ULL+series_count*2ULL);const double expected=expected_score(live_rating,archive.rating);
                const double leg1=score_for_side(paired.legs[0].winner,0),leg2=score_for_side(paired.legs[1].winner,1);update_team(s.variants,live,paired.first_score,expected,leg1,leg2);
                add_contributions(s.variants,live,paired,true);
                top_wins+=paired.legs[0].winner==Winner::Red;bottom_wins+=paired.legs[0].winner==Winner::Blue;leg_draws+=paired.legs[0].winner==Winner::Tie;top_wins+=paired.legs[1].winner==Winner::Red;bottom_wins+=paired.legs[1].winner==Winner::Blue;leg_draws+=paired.legs[1].winner==Winner::Tie;
            }else{
                if(cross_match)++cross;else ++balanced;const Partition p=choose_partition(rng,s.variants,cross_match);const double first_rating=team_rating(s.variants,p.first),second_rating=team_rating(s.variants,p.second);const auto paired=paired_series(team_genomes(s.variants,p.first),team_genomes(s.variants,p.second),p.first,p.second,o.seed+season*1000003ULL+series_count*2ULL);const double expected=expected_score(first_rating,second_rating);const double leg1=score_for_side(paired.legs[0].winner,0),leg2=score_for_side(paired.legs[1].winner,1);update_team(s.variants,p.first,paired.first_score,expected,leg1,leg2);update_team(s.variants,p.second,1-paired.first_score,1-expected,1-leg1,1-leg2);
                add_contributions(s.variants,p.first,paired,true);add_contributions(s.variants,p.second,paired,false);
                top_wins+=paired.legs[0].winner==Winner::Red;bottom_wins+=paired.legs[0].winner==Winner::Blue;leg_draws+=paired.legs[0].winner==Winner::Tie;top_wins+=paired.legs[1].winner==Winner::Red;bottom_wins+=paired.legs[1].winner==Winner::Blue;leg_draws+=paired.legs[1].winner==Winner::Tie;
            }
            ++series_count;
        }
        const auto objective_order=objective_rank(s.variants),elo_order=elo_rank(s.variants);const GenomeLineup champion=lineup_from_order(s.variants,objective_order);s.last_elo_champion=lineup_from_order(s.variants,elo_order);double champion_rating=0;for(int i=0;i<4;++i)champion_rating+=s.variants[objective_order[i]].elo;champion_rating/=4;
        GenomeLineup candidate_clones{},incumbent_clones{};candidate_clones.fill(s.variants[objective_order[0]].genome);incumbent_clones.fill(s.base);
        const Bench candidate_gate=benchmark_team(candidate_clones,s.hof.front().genomes,30,930000);
        const Bench incumbent_gate=benchmark_team(incumbent_clones,s.hof.front().genomes,30,930000);
        const bool promoted=candidate_gate.champion_caps>incumbent_gate.champion_caps+1e-12||(std::abs(candidate_gate.champion_caps-incumbent_gate.champion_caps)<1e-12&&candidate_gate.champion_wins>=incumbent_gate.champion_wins);
        if(promoted)s.base=s.variants[objective_order[0]].genome;
        Bench bench{};const bool did_bench=o.benchmark_every>0&&(season%o.benchmark_every==0||season==o.seasons-1);if(did_bench)bench=benchmark_team(champion,s.hof.front().genomes,o.benchmark_trials,900000);
        const int best_objective=100*s.variants[objective_order[0]].season_returns+25*s.variants[objective_order[0]].season_pickups;
        std::ostringstream line;line<<season<<','<<best_objective<<','<<s.variants[objective_order[0]].season_returns<<','<<s.variants[objective_order[0]].season_pickups<<','<<s.variants[elo_order[0]].elo<<','<<mean_rating(s.variants)<<','<<series_count<<','<<balanced<<','<<cross<<','<<hof_count<<','<<top_wins<<','<<bottom_wins<<','<<leg_draws<<','<<promoted<<','<<candidate_gate.champion_wins<<','<<incumbent_gate.champion_wins<<','<<candidate_gate.champion_caps<<','<<incumbent_gate.champion_caps<<',';
        if(did_bench)line<<bench.champion_wins<<','<<bench.anchor_wins<<','<<bench.draws<<','<<bench.champion_caps<<','<<bench.anchor_caps;else line<<",,,,";s.history.push_back(line.str());
        std::cout<<"season "<<std::setw(6)<<season<<std::fixed<<std::setprecision(1)<<" objective="<<best_objective<<" events="<<s.variants[objective_order[0]].season_returns<<'/'<<s.variants[objective_order[0]].season_pickups<<" elo="<<s.variants[elo_order[0]].elo<<" mean="<<mean_rating(s.variants)<<" promoted="<<promoted<<" gate="<<std::setprecision(2)<<candidate_gate.champion_caps<<'/'<<incumbent_gate.champion_caps<<" series="<<series_count<<" mix="<<balanced<<'/'<<cross<<'/'<<hof_count<<" sides="<<top_wins<<'/'<<bottom_wins<<'/'<<leg_draws;
        if(did_bench)std::cout<<" bench="<<bench.champion_wins<<'/'<<bench.anchor_wins<<'/'<<bench.draws<<" caps="<<std::setprecision(2)<<bench.champion_caps<<'/'<<bench.anchor_caps;
        std::cout<<'\n';
        s.hof.push_back({champion,champion_rating,season});if(s.hof.size()>20)s.hof.erase(s.hof.begin()+1);reproduce(rng,s,o);
        if(o.checkpoint_every>0&&((season+1)%o.checkpoint_every==0||season==o.seasons-1))save(o,s,season+1);
        if(o.max_hours>0){const std::chrono::duration<double,std::ratio<3600>>elapsed=std::chrono::steady_clock::now()-started;if(elapsed.count()>=o.max_hours){save(o,s,season+1);std::cout<<"wall-clock limit reached at season "<<season+1<<" after "<<std::setprecision(4)<<elapsed.count()<<" hours\n";break;}}
    }
    save(o,s,-1);
}

Options parse(int argc,char**argv){Options o;auto value=[&](int&i,const std::string&x){if(++i>=argc)throw std::runtime_error(x+" requires value");return std::string(argv[i]);};for(int i=1;i<argc;++i){std::string x=argv[i];if(x=="--seed-team")o.seed_team=value(i,x);else if(x=="--games")o.games=std::stoi(value(i,x));else if(x=="--seasons")o.seasons=std::stoi(value(i,x));else if(x=="--elite")o.elite=std::stoi(value(i,x));else if(x=="--mut")o.mutation=std::stod(value(i,x));else if(x=="--seed")o.seed=std::stoull(value(i,x));else if(x=="--checkpoint-every")o.checkpoint_every=std::stoi(value(i,x));else if(x=="--benchmark-every")o.benchmark_every=std::stoi(value(i,x));else if(x=="--benchmark-trials")o.benchmark_trials=std::stoi(value(i,x));else if(x=="--max-hours")o.max_hours=std::stod(value(i,x));else if(x=="--out")o.out=value(i,x);else if(x=="--quick"){o.games=4;o.seasons=2;o.benchmark_every=1;o.benchmark_trials=10;}else throw std::runtime_error("unknown option "+x);}if(o.seed_team.empty())throw std::runtime_error("--seed-team is required");if(o.elite<1||o.elite>8||o.games<2)throw std::runtime_error("invalid elite or games");return o;}

} // namespace scrimmage

#ifndef SCRIMMAGE_NO_MAIN
int main(int argc,char**argv){try{scrimmage::run(scrimmage::parse(argc,argv));return 0;}catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<'\n';return 1;}}
#endif
