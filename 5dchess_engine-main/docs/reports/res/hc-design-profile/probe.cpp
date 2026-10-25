#include "hypercuboid.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <algorithm>
using Clock=std::chrono::steady_clock;
double ms(Clock::duration d){return std::chrono::duration<double,std::milli>(d).count();}
int main(int argc,char**argv){
 std::string path=argv[1], mode=argv[2], partition=argc>4?argv[4]:"all"; double budget=std::stod(argv[3]);
 std::ifstream f(path);std::ostringstream text;text<<f.rdbuf(); state s=state::from_pgn(text.str());
 auto begin=Clock::now();auto [info,space]=HC_info::build_HC(s);double build=ms(Clock::now()-begin);
 if(partition=="nonbranch"){HC hc=std::move(space.back());space=search_space{};space.push_back(std::move(hc));}
 if(partition=="restricted"){
  space.pop_back();auto [a,b]=s.get_initial_lines_range();auto [l,r]=s.get_lines_range();
  int advantage=s.get_present().c()?((r-b)-(a-l)):((a-l)-(r-b));
  for(HC &hc:space){int last=std::min(int(info.first_branch_axis())+advantage+1,int(info.dimension())-1);
   for(int n=info.first_branch_axis();n<=last;++n)hc[n].erase_if([&](int i){auto sm=info.get_semimove(n,i);return sm.is<arriving_move>() && sm.get<arriving_move>().m.to.t()<s.get_present().t();});
  }
 }
 bool global=mode.find("global")==0,full=mode.find("full")!=std::string::npos,inplace=mode.find("inplace")!=std::string::npos;
 size_t attempts=0,empty=0,problems=0,visits=0,cuts=0,skips=0,peak=space.size();double point_ms=0,problem_ms=0,remove_ms=0;std::string result="exhausted";
 auto start=Clock::now();
 while(!space.empty()){
  if(ms(Clock::now()-start)>=budget){result="timeout";break;}
  ++attempts;HC hc=std::move(space.back());space.pop_back();auto t=Clock::now();auto p=info.take_point(hc);point_ms+=ms(Clock::now()-t);
  if(!p){++empty;continue;}
  t=Clock::now();auto problem=info.find_problem(*p,global?info.get_universe():hc);problem_ms+=ms(Clock::now()-t);
  if(!problem){result="action";auto action=info.to_action(*p);if(!s.can_apply(action::from_moveseq(action,s)))return 2;break;}
  ++problems;t=Clock::now();
  if(!full||inplace){space.push_back(std::move(hc));space.profile_remove(*problem,full,visits,cuts,skips);}
  else{
   search_space next;
   for(const HC &other:space){++visits;if(other.intersects(*problem))++cuts;next.concat(other.remove_slice_carefully(*problem));}
   ++visits;++cuts;
   next.concat(global?hc.remove_slice_carefully(*problem):hc.remove_slice(*problem));space=std::move(next);
  }
  remove_ms+=ms(Clock::now()-t);peak=std::max(peak,space.size());
 }
 std::cout<<path<<','<<partition<<','<<mode<<','<<result<<','<<ms(Clock::now()-start)<<','<<build<<','<<point_ms<<','<<problem_ms<<','<<remove_ms<<','<<attempts<<','<<empty<<','<<problems<<','<<visits<<','<<cuts<<','<<skips<<','<<peak<<','<<space.size()<<'\n';
}
