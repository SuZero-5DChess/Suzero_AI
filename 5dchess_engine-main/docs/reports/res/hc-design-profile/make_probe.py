from pathlib import Path
p=Path('/tmp/hc-design-profile')
h=Path('src/misc/geometry.h').read_text().replace('    void prune_empty();','    void prune_empty();\n    void profile_remove(const slice &, bool full, size_t &visits, size_t &cuts, size_t &skips);')
(p/'geometry.h').write_text(h)
c=Path('src/misc/geometry.cpp').read_text()+'''
void search_space::profile_remove(const slice &s, bool full, size_t &visits, size_t &cuts, size_t &skips) {
 if(hcs.empty()) return;
 size_t intersections=0,disjoint=0;
 auto current=std::prev(hcs.end()); bool force=true;
 while(true){
  ++visits;
  bool has_previous=current!=hcs.begin(); auto previous=current;
  if(has_previous)--previous;
  if(current->intersects(s)){
   auto after=std::next(current);
   bool good=full || force || current->is_slice_good(s,1);
   if(good)++cuts;else ++skips;
   auto pieces=(full||force)?current->remove_slice_carefully(s):current->remove_slice_if_good(s,1);
   hcs.erase(current); hcs.splice(after,pieces.hcs); ++intersections;
  }else ++disjoint;
  if(!has_previous || (!full && disjoint*10>=intersections))break;
  force=false;current=previous;
 }
}
'''
(p/'geometry.cpp').write_text(c)
