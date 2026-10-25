#include "profile.h"
#include "finetree.h"
#include "rollout_policy.h"
#include <fstream>
#include <sstream>
#include <iostream>
int main(int argc,char**argv){
 std::ifstream f(argv[1]);std::ostringstream text;text<<f.rdbuf();auto s=state::from_pgn(text.str());std::string mode=argv[2];profile::budget=std::stod(argv[3]);unsigned seed=std::stoul(argv[4]);std::mt19937 rng(seed);std::string result="exhausted";
 profile::start=profile::Clock::now();
 try{
  if(mode=="random"||mode=="weighted"){
   auto action=mode=="random"?random_action_selection{}(s,{},&rng):weighted_action_selection{}(s,{},&rng);
   if(action){result="action";if(!s.can_apply(action::from_moveseq(*action,s)))return 2;}
  }else{
   auto root=profile::timed(profile::setup_ms, [&]{return fine_node<>::make_root(s);});
   bool found=mode=="fine-natural"?bool(root->search(natural_HC_ordering{}).first()):bool(root->search(random_HC_ordering(root->get_hc_info().get_universe(),rng)).first());
   if(found)result="action";
   profile::timed(profile::destroy_ms, [&]{root.reset();});
  }
 }catch(const std::runtime_error &e){if(std::string(e.what())!="budget")throw;result="timeout";}
 std::cout<<argv[1]<<','<<mode<<','<<seed<<','<<result<<','<<profile::ms(profile::Clock::now()-profile::start)<<','<<profile::point_ms<<','<<profile::problem_ms<<','<<profile::remove_ms<<','<<profile::attempts<<','<<profile::empty<<','<<profile::problems<<','<<profile::setup_ms<<','<<profile::isolate_ms<<','<<profile::normalize_ms<<','<<profile::destroy_ms<<'\n';
}
