#include "hypercuboid.h"
#include "rollout_policy.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>
#include <cmath>
int main(){
 std::ifstream f("test/pgn/100timelines.5dpgn");std::ostringstream text;text<<f.rdbuf();auto s=state::from_pgn(text.str());
 using clock=std::chrono::steady_clock;
 auto ms=[](auto d){return std::chrono::duration<double,std::milli>(d).count();};
 std::cout<<"trial,build_ms,coordinate_weights_ms,ordering_ms,coordinates\n";
 for(int trial=0;trial<3;++trial){
  auto start=clock::now();auto [info,space]=HC_info::build_HC(s);auto built=clock::now();
  weighted_action_selection policy;move_evaluator evaluator(info,default_move_evaluation_weights);
  std::vector<std::vector<float>> coordinate_weights(info.get_universe().dimension());size_t coordinates=0;
  for(index_t axis=0;axis<info.get_universe().dimension();++axis){
   auto &w=coordinate_weights[axis];w.reserve(info.get_universe()[axis].size());
   for(index_t coordinate:info.get_universe()[axis]){w.push_back(std::exp(evaluator.score(axis,coordinate)/policy.get_temperature()));++coordinates;}
  }
  auto scored=clock::now();std::mt19937 rng(42);weighted_HC_ordering order(info.get_universe(),std::move(coordinate_weights),rng);auto ordered=clock::now();
  std::cout<<trial<<','<<ms(built-start)<<','<<ms(scored-built)<<','<<ms(ordered-scored)<<','<<coordinates<<'\n';
 }
}
