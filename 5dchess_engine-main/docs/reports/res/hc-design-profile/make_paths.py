from pathlib import Path
import subprocess,shutil
p=Path('/tmp/hc-design-profile')
for version in ['old','global-adaptive','new']:
 d=p/version
 shutil.copytree('src',d/'src',dirs_exist_ok=True)
 if version!='new':
  for name in ['core/finetree.h','core/finetree.inl','engine/rollout_policy.cpp','engine/default_policy.h']:
   (d/'src'/name).write_bytes(subprocess.check_output(['git','show','b6e16ec58efd6a0c865fa0afd808f170b48794d1:src/'+name]))
 for name in ['core/finetree.inl','engine/rollout_policy.cpp']:
  f=d/'src'/name;s=f.read_text()
  if version=='global-adaptive':
   s=s.replace('hc_info.find_problem(*pt_opt, hc)','hc_info.find_problem(*pt_opt, hc_info.get_universe())').replace('hc_info.find_problem(*point, hc)','hc_info.find_problem(*point, hc_info.get_universe())').replace('hc.remove_slice(*problem)','hc.remove_slice_carefully(*problem)')
  s='#include "profile.h"\n'+s
  if 'finetree' in name:
   s=s.replace('auto pt_opt = hc_info.take_point(hc, order);','profile::check(); ++profile::attempts;\n        auto pt_opt = profile::timed(profile::point_ms, [&]{return hc_info.take_point(hc, order);});\n        if(!pt_opt) ++profile::empty;')
   for target in ['hc','hc_info.get_universe()']:
    s=s.replace('auto problem = hc_info.find_problem(*pt_opt, '+target+');','auto problem = profile::timed(profile::problem_ms, [&]{return hc_info.find_problem(*pt_opt, '+target+');});')
   for arg in ['*problem, cell','*problem']:
    s=s.replace('            remove_problem('+arg+');','            ++profile::problems; profile::timed(profile::remove_ms, [&]{remove_problem('+arg+');});')
  else:
   s=s.replace('auto point = hc_info.take_point(hc, order);','profile::check(); ++profile::attempts;\n        auto point = profile::timed(profile::point_ms, [&]{return hc_info.take_point(hc, order);});\n        if(!point) ++profile::empty;')
   # Scope timers cover whole if branches: original uses if-init while new has standalone variable.
   s=s.replace('hc_info.find_problem(*point, hc)','profile::timed(profile::problem_ms, [&]{return hc_info.find_problem(*point, hc);})')
   s=s.replace('hc_info.find_problem(*point, hc_info.get_universe())','profile::timed(profile::problem_ms, [&]{return hc_info.find_problem(*point, hc_info.get_universe());})')
   s=s.replace('space.concat(hc.remove_slice(*problem));','++profile::problems; profile::timed(profile::remove_ms, [&]{space.concat(hc.remove_slice(*problem));});')
   s=s.replace('space.concat(hc.remove_slice_carefully(*problem));','++profile::problems; profile::timed(profile::remove_ms, [&]{space.concat(hc.remove_slice_carefully(*problem));});')
   s=s.replace('space.remove_slice_backwards(*problem, true, 10);','++profile::problems; profile::timed(profile::remove_ms, [&]{space.remove_slice_backwards(*problem, true, 10);});')
   s=s.replace('search_space remaining;','++profile::problems; auto profile_remove_start = profile::Clock::now();\n        search_space remaining;')
   s=s.replace('space = std::move(remaining);','space = std::move(remaining); profile::remove_ms += profile::ms(profile::Clock::now()-profile_remove_start);')
  if 'finetree' in name:
   s=s.replace('fine_node<T> *final_node = isolate(pt, cell, hc);','fine_node<T> *final_node = profile::timed(profile::isolate_ms, [&]{return isolate(pt, cell, hc);});').replace('fine_node<T> *next_node = normalize(pt, cell, final_node);','fine_node<T> *next_node = profile::timed(profile::normalize_ms, [&]{return normalize(pt, cell, final_node);});')
  f.write_text(s)
