import 'package:flutter/material.dart';

void main() => runApp(const OtbokhihaApp());

class Recipe {
  final String id, name, country, category;
  final int mins, baseServings, calories;
  final List<String> ingredients, steps, tips;
  const Recipe(this.id, this.name, this.country, this.category, this.mins, this.baseServings, this.calories, this.ingredients, this.steps, this.tips);
}

const recipes = <Recipe>[
  Recipe('koshary','كشري مصري','مصر','رئيسي',55,4,620,['أرز','عدس','مكرونة','حمص','طماطم','بصل','ثوم'],['اسلقي العدس.','اطهي الأرز.','اسلقي المكرونة.','حضري الصلصة والبصل المقرمش.','قدمي الطبقات معًا.'],['نشفي البصل قبل التحمير.','لا تفرطي في سلق العدس.']),
  Recipe('potato_chicken','صينية بطاطس بالفراخ','مصر','فراخ',70,4,510,['فراخ','بطاطس','طماطم','بصل','ثوم'],['تبلي الفراخ.','قطعي الخضار.','رصي المكونات.','غطي ثم حمري آخر التسوية.'],['وحدي سمك شرائح البطاطس.','قللي الزيت لو عايزة نسخة أخف.']),
  Recipe('molokhia','ملوخية مصرية','مصر','خضار',35,4,220,['ملوخية','مرق','ثوم','كزبرة'],['سخني المرق.','أضيفي الملوخية دون غليان قوي.','أضيفي التقلية.'],['الغليان الشديد يغير القوام.']),
  Recipe('lentil','شوربة عدس','مصر','شوربة',35,4,280,['عدس','جزر','طماطم','بصل'],['اسلقي العدس والخضار.','اخلطي.','تبلي واتركي غلوة قصيرة.'],['أضيفي الماء تدريجيًا.']),
  Recipe('rice_milk','أرز باللبن','مصر','حلويات',45,6,320,['أرز','لبن','سكر','مكسرات'],['اطهي الأرز.','أضيفي اللبن تدريجيًا.','أضيفي السكر حتى يثقل القوام.'],['قلبي باستمرار.']),
  Recipe('kabsa','كبسة دجاج','السعودية','أرز',75,6,680,['أرز بسمتي','فراخ','طماطم','بصل','بهارات كبسة'],['شوحي البصل والبهارات.','اطهي الدجاج.','استخدمي المرق لتسوية الأرز.'],['انقعي الأرز قبل التسوية.']),
  Recipe('jareesh','جريش','السعودية','حبوب',90,5,430,['جريش','لبن','بصل','فراخ'],['انقعي الجريش.','اطهيه ببطء.','أضيفي اللبن.'],['التسوية الهادئة تعطي قوامًا كريميًا.']),
  Recipe('mansaf','منسف أردني','الأردن','لحوم',120,6,780,['لحم','أرز','جميد','خبز شراك','مكسرات'],['اطهي اللحم.','حضري الجميد.','اطهي الأرز.','رتبي وقدمي.'],['اضبطي ملوحة الجميد أولًا.']),
  Recipe('maqluba','مقلوبة','الأردن','أرز',85,5,590,['أرز','فراخ','باذنجان','بطاطس','طماطم'],['حضري الخضار.','رتبي المكونات.','سوي على نار هادئة.','اتركيها ترتاح ثم اقلبيها.'],['الراحة قبل القلب تقلل التفكك.']),
  Recipe('fattoush','فتوش','لبنان','سلطات',20,4,180,['خس','طماطم','خيار','خبز','سمسم'],['قطعي الخضار.','حمصي الخبز.','اخلطي التتبيلة وقدمي.'],['أضيفي الخبز قبل التقديم مباشرة.']),
  Recipe('shakriya','شاكرية بالدجاج','سوريا','فراخ',65,4,520,['فراخ','لبن','أرز','ثوم'],['اسلقي الدجاج.','حضري اللبن على نار هادئة.','أضيفي الدجاج وقدمي مع الأرز.'],['حرارة هادئة وتحريك مستمر.']),
  Recipe('tagine','طاجين خضار مغربي','المغرب','خضار',70,4,310,['بطاطس','جزر','كوسة','طماطم','بصل'],['رتبي الخضار.','أضيفي التوابل والطماطم.','سوي ببطء.'],['التسوية البطيئة تركز النكهة.']),
  Recipe('pasta','مكرونة بصلصة الطماطم','إيطاليا','مكرونة',30,4,410,['مكرونة','طماطم','ثوم','بصل'],['اسلقي المكرونة.','حضري الصلصة.','اخلطي قبل التقديم.'],['احتفظي بقليل من ماء السلق.']),
  Recipe('aloo','ألو ماتر','الهند','خضار',40,4,330,['بطاطس','بسلة','طماطم','بصل'],['شوحي البصل والتوابل.','أضيفي الطماطم والبطاطس.','أضيفي البسلة واتركيها تنضج.'],['ابدئي بتوابل قليلة ثم زودي.']),
];

const problems = <String,List<String>>{
  'الأكل ملحه زاد':['زودي مكونات غير مملحة إن أمكن.','في الشوربة أضيفي سائلًا غير مملح تدريجيًا.','قدميه مع أرز أو خبز قليل الملح.'],
  'الأرز معجن':['ارفعيه من النار فورًا.','افرديه في صينية ليخرج البخار.','تجنبي التقليب العنيف.'],
  'الأرز ناشف':['أضيفي قليل ماء ساخن.','غطيه على أهدأ نار عدة دقائق.'],
  'البشاميل كلكع':['ارفعيه من النار واخفقي.','أضيفي اللبن تدريجيًا.'],
  'الفراخ نشفت':['قدميها مع صوص أو مرق.','راقبي وقت التسوية في المرة القادمة.'],
  'الكيكة هبطت':['لا تفتحي الفرن مبكرًا.','راجعي صلاحية البيكنج باودر.'],
  'الصوص خفيف':['اتركيه يتسبك بدون غطاء.','استخدمي قليل نشا مذاب في ماء بارد إذا كان مناسبًا.'],
  'الأكل حار زيادة':['زودي باقي مكونات الوصفة غير الحارة.','الزبادي قد يساعد في وصفات مناسبة.'],
};

class AppState extends ChangeNotifier {
  String familyName = 'أسرتي';
  int members = 4;
  final Set<String> allergies = {};
  final Set<String> disliked = {};
  final Set<String> favorites = {};
  List<Recipe> plan = [];
  final Set<String> bought = {};

  bool safe(Recipe r) => r.ingredients.every((x) => !allergies.contains(x) && !disliked.contains(x));
  List<Recipe> get safeRecipes => recipes.where(safe).toList();
  void setFamily(String name, int count, Set<String> a, Set<String> d) { familyName=name; members=count; allergies..clear()..addAll(a); disliked..clear()..addAll(d); notifyListeners(); }
  void toggleFav(String id){favorites.contains(id)?favorites.remove(id):favorites.add(id); notifyListeners();}
  void makePlan(int days){final p=safeRecipes; plan=List.generate(days,(i)=>p[i%p.length]); bought.clear(); notifyListeners();}
  void swap(int i){final p=safeRecipes;if(p.length<2)return; final n=(p.indexWhere((x)=>x.id==plan[i].id)+1)%p.length;plan[i]=p[n];notifyListeners();}
  Map<String,int> get shopping {final m=<String,int>{};for(final r in plan){for(final x in r.ingredients){m[x]=(m[x]??0)+1;}}return m;}
}

class OtbokhihaApp extends StatefulWidget { const OtbokhihaApp({super.key}); @override State<OtbokhihaApp> createState()=>_OtState(); }
class _OtState extends State<OtbokhihaApp>{final s=AppState(); bool configured=false;
 @override Widget build(BuildContext c)=>AnimatedBuilder(animation:s,builder:(_,__)=>MaterialApp(debugShowCheckedModeBanner:false,locale:const Locale('ar'),theme:ThemeData(useMaterial3:true,colorSchemeSeed:const Color(0xFFF28C55),scaffoldBackgroundColor:const Color(0xFFFFFAF5)),home:Directionality(textDirection:TextDirection.rtl,child:configured?Shell(s:s):Setup(s:s,onDone:(){setState(()=>configured=true);}))));}

class Setup extends StatefulWidget{final AppState s;final VoidCallback onDone;const Setup({super.key,required this.s,required this.onDone});@override State<Setup> createState()=>_SetupState();}
class _SetupState extends State<Setup>{final name=TextEditingController(text:'أسرتي');int n=4;final a=<String>{},d=<String>{}; final allerg=['فول سوداني','مكسرات','لبن','بيض','قمح','سمسم','سمك','جمبري']; final dislike=['بامية','سمك','باذنجان','فلفل حار','ثوم','بصل'];
 Widget chips(List<String> l,Set<String> s)=>Wrap(spacing:7,children:l.map((x)=>FilterChip(label:Text(x),selected:s.contains(x),onSelected:(v)=>setState(()=>v?s.add(x):s.remove(x)))).toList());
 @override Widget build(BuildContext c)=>Scaffold(appBar:AppBar(title:const Text('اطبخيها - إعداد الأسرة')),body:ListView(padding:const EdgeInsets.all(18),children:[const Text('نسخة تجريبية',style:TextStyle(fontSize:28,fontWeight:FontWeight.bold)),const Text('خلّي المقترحات والخطة مناسبة لبيتك.'),const SizedBox(height:18),TextField(controller:name,decoration:const InputDecoration(labelText:'اسم الأسرة',border:OutlineInputBorder())),const SizedBox(height:12),Row(children:[const Expanded(child:Text('عدد أفراد الأسرة')),IconButton(onPressed:n>1?()=>setState(()=>n--):null,icon:const Icon(Icons.remove_circle_outline)),Text('$n',style:const TextStyle(fontSize:22)),IconButton(onPressed:()=>setState(()=>n++),icon:const Icon(Icons.add_circle_outline))]),const SizedBox(height:12),const Text('حساسية / ممنوع',style:TextStyle(fontWeight:FontWeight.bold)),chips(allerg,a),const SizedBox(height:12),const Text('غير مفضل',style:TextStyle(fontWeight:FontWeight.bold)),chips(dislike,d),const SizedBox(height:12),const Card(child:Padding(padding:EdgeInsets.all(12),child:Text('تنبيه: بيانات الحساسية والسعرات في النسخة التجريبية إرشادية وليست بديلًا عن التحقق الطبي أو قراءة ملصقات المنتجات.'))),FilledButton(onPressed:(){widget.s.setFamily(name.text.trim().isEmpty?'أسرتي':name.text.trim(),n,a,d);widget.onDone();},child:const Text('ابدئي التجربة'))]));}

class Shell extends StatefulWidget{final AppState s;const Shell({super.key,required this.s});@override State<Shell> createState()=>_ShellState();}
class _ShellState extends State<Shell>{int i=0;@override Widget build(BuildContext c){final p=[Home(s:widget.s,go:(x)=>setState(()=>i=x)),Discover(s:widget.s),Plan(s:widget.s),Shopping(s:widget.s),Profile(s:widget.s)];return Scaffold(body:IndexedStack(index:i,children:p),bottomNavigationBar:NavigationBar(selectedIndex:i,onDestinationSelected:(x)=>setState(()=>i=x),destinations:const [NavigationDestination(icon:Icon(Icons.home_outlined),label:'الرئيسية'),NavigationDestination(icon:Icon(Icons.explore_outlined),label:'اكتشفي'),NavigationDestination(icon:Icon(Icons.calendar_month_outlined),label:'خطتي'),NavigationDestination(icon:Icon(Icons.shopping_cart_outlined),label:'مشترياتي'),NavigationDestination(icon:Icon(Icons.person_outline),label:'حسابي')]));}}

class Home extends StatelessWidget{final AppState s;final void Function(int) go;const Home({super.key,required this.s,required this.go});@override Widget build(BuildContext c)=>Scaffold(appBar:AppBar(title:Text('أهلاً، ${s.familyName} 👋')),body:ListView(padding:const EdgeInsets.all(14),children:[const Text('هناكل إيه النهارده؟',style:TextStyle(fontSize:28,fontWeight:FontWeight.bold)),Text('مناسب لأسرة من ${s.members} أفراد'),const SizedBox(height:14),GridView.count(shrinkWrap:true,physics:const NeverScrollableScrollPhysics(),crossAxisCount:2,childAspectRatio:1.25,children:[card(c,Icons.public,'أكلات العالم',()=>Navigator.push(c,MaterialPageRoute(builder:(_)=>World(s:s)))),card(c,Icons.auto_awesome,'مقترحات',()=>Navigator.push(c,MaterialPageRoute(builder:(_)=>Suggest(s:s)))),card(c,Icons.sos,'مشكلات وحلول',()=>Navigator.push(c,MaterialPageRoute(builder:(_)=>const Problems()))),card(c,Icons.calendar_month,'خطتي',()=>go(2)),card(c,Icons.shopping_cart,'مشترياتي',()=>go(3)),card(c,Icons.favorite,'المفضلة',()=>Navigator.push(c,MaterialPageRoute(builder:(_)=>Favorites(s:s))))]),const SizedBox(height:18),Text('مناسب لأسرتك (${s.safeRecipes.length})',style:const TextStyle(fontSize:20,fontWeight:FontWeight.bold)),...s.safeRecipes.take(5).map((r)=>RecipeTile(r:r,s:s))]));
 Widget card(BuildContext c,IconData i,String t,VoidCallback f)=>Card(child:InkWell(onTap:f,child:Padding(padding:const EdgeInsets.all(14),child:Column(mainAxisAlignment:MainAxisAlignment.center,children:[Icon(i,size:30),const SizedBox(height:6),Text(t,style:const TextStyle(fontWeight:FontWeight.bold))]))));}

class Discover extends StatefulWidget{final AppState s;const Discover({super.key,required this.s});@override State<Discover> createState()=>_DiscoverState();}
class _DiscoverState extends State<Discover>{String q='';@override Widget build(BuildContext c){final l=widget.s.safeRecipes.where((r)=>q.isEmpty||r.name.contains(q)||r.country.contains(q)||r.ingredients.any((x)=>x.contains(q))).toList();return Scaffold(appBar:AppBar(title:const Text('اكتشفي')),body:Column(children:[Padding(padding:const EdgeInsets.all(12),child:TextField(onChanged:(v)=>setState(()=>q=v),decoration:const InputDecoration(prefixIcon:Icon(Icons.search),hintText:'أكلة، مكوّن، بلد...',border:OutlineInputBorder()))),Expanded(child:ListView(children:l.map((r)=>RecipeTile(r:r,s:widget.s)).toList()))]));}}

class World extends StatelessWidget{final AppState s;const World({super.key,required this.s});@override Widget build(BuildContext c){final countries=s.safeRecipes.map((x)=>x.country).toSet().toList();return Scaffold(appBar:AppBar(title:const Text('أكلات العالم')),body:ListView(children:countries.map((x)=>ExpansionTile(title:Text(x,style:const TextStyle(fontWeight:FontWeight.bold)),children:s.safeRecipes.where((r)=>r.country==x).map((r)=>RecipeTile(r:r,s:s)).toList())).toList()));}}

class Suggest extends StatefulWidget{final AppState s;const Suggest({super.key,required this.s});@override State<Suggest> createState()=>_SuggestState();}
class _SuggestState extends State<Suggest>{final c=TextEditingController();List<Recipe> l=[];@override void initState(){super.initState();l=widget.s.safeRecipes;}@override Widget build(BuildContext x)=>Scaffold(appBar:AppBar(title:const Text('اقترحلي أكلة')),body:ListView(padding:const EdgeInsets.all(14),children:[TextField(controller:c,maxLines:2,decoration:const InputDecoration(hintText:'مثال: بطاطس، فراخ، طماطم',border:OutlineInputBorder())),const SizedBox(height:8),FilledButton(onPressed:(){final t=c.text.split(RegExp(r'[,،\n]')).map((e)=>e.trim()).where((e)=>e.isNotEmpty).toSet();setState(()=>l=widget.s.safeRecipes.where((r)=>r.ingredients.any((x)=>t.any((y)=>x.contains(y)||y.contains(x)))).toList());},child:const Text('اقترحلي')),Text('النتائج: ${l.length}'),...l.map((r)=>RecipeTile(r:r,s:widget.s))]));}

class Problems extends StatelessWidget{const Problems({super.key});@override Widget build(BuildContext c)=>Scaffold(appBar:AppBar(title:const Text('مشكلات وحلول')),body:ListView(children:problems.entries.map((e)=>Card(child:ExpansionTile(title:Text(e.key,style:const TextStyle(fontWeight:FontWeight.bold)),children:e.value.map((x)=>ListTile(leading:const Icon(Icons.check_circle_outline),title:Text(x))).toList()))).toList()));}

class Plan extends StatelessWidget{final AppState s;const Plan({super.key,required this.s});@override Widget build(BuildContext c)=>Scaffold(appBar:AppBar(title:const Text('خطتي')),body:ListView(padding:const EdgeInsets.all(12),children:[Row(children:[Expanded(child:FilledButton(onPressed:()=>s.makePlan(7),child:const Text('خطة أسبوع'))),const SizedBox(width:8),Expanded(child:OutlinedButton(onPressed:()=>s.makePlan(30),child:const Text('خطة شهر')))]),const SizedBox(height:8),Text('الخطة تراعي ${s.members} أفراد والتفضيلات المسجلة.'),...s.plan.asMap().entries.map((e)=>Card(child:ListTile(title:Text('اليوم ${e.key+1}'),subtitle:Text(e.value.name),trailing:IconButton(onPressed:()=>s.swap(e.key),icon:const Icon(Icons.refresh)))))]));}

class Shopping extends StatelessWidget{final AppState s;const Shopping({super.key,required this.s});@override Widget build(BuildContext c){final m=s.shopping;return Scaffold(appBar:AppBar(title:const Text('مشترياتي')),body:m.isEmpty?const Center(child:Text('اعملي خطة أولًا علشان تتجمع المشتريات.')):ListView(children:m.entries.map((e)=>CheckboxListTile(value:s.bought.contains(e.key),onChanged:(_){s.bought.contains(e.key)?s.bought.remove(e.key):s.bought.add(e.key);s.notifyListeners();},title:Text(e.key),subtitle:Text('مطلوب للوجبات: ${e.value} مرة'))).toList()));}}

class Favorites extends StatelessWidget{final AppState s;const Favorites({super.key,required this.s});@override Widget build(BuildContext c){final l=recipes.where((r)=>s.favorites.contains(r.id)).toList();return Scaffold(appBar:AppBar(title:const Text('المفضلة')),body:l.isEmpty?const Center(child:Text('لسه مفيش وصفات محفوظة.')):ListView(children:l.map((r)=>RecipeTile(r:r,s:s)).toList()));}}

class Profile extends StatelessWidget{final AppState s;const Profile({super.key,required this.s});@override Widget build(BuildContext c)=>Scaffold(appBar:AppBar(title:const Text('حسابي وأسرتي')),body:ListView(padding:const EdgeInsets.all(16),children:[ListTile(leading:const Icon(Icons.family_restroom),title:Text(s.familyName),subtitle:Text('${s.members} أفراد')),ListTile(title:const Text('الحساسية / الممنوع'),subtitle:Text(s.allergies.isEmpty?'لا يوجد':s.allergies.join('، '))),ListTile(title:const Text('غير المفضل'),subtitle:Text(s.disliked.isEmpty?'لا يوجد':s.disliked.join('، '))),const Card(child:Padding(padding:EdgeInsets.all(12),child:Text('نسخة MVP تجريبية. السعرات والمعلومات الصحية تقريبية.')))]));}

class RecipeTile extends StatelessWidget{final Recipe r;final AppState s;const RecipeTile({super.key,required this.r,required this.s});@override Widget build(BuildContext c)=>Card(child:ListTile(title:Text(r.name,style:const TextStyle(fontWeight:FontWeight.bold)),subtitle:Text('${r.country} • ${r.mins} دقيقة • ${r.calories} kcal تقريبًا'),trailing:Icon(s.favorites.contains(r.id)?Icons.favorite:Icons.chevron_left),onTap:()=>Navigator.push(c,MaterialPageRoute(builder:(_)=>RecipePage(r:r,s:s)))));}

class RecipePage extends StatefulWidget{final Recipe r;final AppState s;const RecipePage({super.key,required this.r,required this.s});@override State<RecipePage> createState()=>_RecipePageState();}
class _RecipePageState extends State<RecipePage>{late int servings;@override void initState(){super.initState();servings=widget.s.members;}@override Widget build(BuildContext c){final r=widget.r;return Scaffold(appBar:AppBar(title:Text(r.name),actions:[IconButton(onPressed:(){widget.s.toggleFav(r.id);setState((){});},icon:Icon(widget.s.favorites.contains(r.id)?Icons.favorite:Icons.favorite_border))]),body:ListView(padding:const EdgeInsets.all(14),children:[Text('${r.country} • ${r.category}',style:const TextStyle(fontSize:18)),Text('${r.mins} دقيقة • ${r.calories} سعر/فرد تقريبًا'),Row(children:[const Text('عدد الأفراد: '),IconButton(onPressed:servings>1?()=>setState(()=>servings--):null,icon:const Icon(Icons.remove_circle_outline)),Text('$servings'),IconButton(onPressed:()=>setState(()=>servings++),icon:const Icon(Icons.add_circle_outline))]),section('المكونات',r.ingredients.map((x)=>ListTile(leading:const Icon(Icons.check),title:Text(x),subtitle:Text('الكمية تُضبط لعدد $servings أفراد في النسخة النهائية'))).toList()),section('الطريقة',r.steps.asMap().entries.map((e)=>ListTile(leading:CircleAvatar(child:Text('${e.key+1}')),title:Text(e.value))).toList()),section('تريكات النجاح',r.tips.map((x)=>ListTile(leading:const Icon(Icons.lightbulb_outline),title:Text(x))).toList()),FilledButton.icon(onPressed:(){widget.s.plan.add(r);widget.s.notifyListeners();ScaffoldMessenger.of(c).showSnackBar(const SnackBar(content:Text('اتضافت للخطة')));},icon:const Icon(Icons.add_task),label:const Text('أضيفي للخطة'))]));}
 Widget section(String t,List<Widget> ch)=>Card(child:ExpansionTile(initiallyExpanded:true,title:Text(t,style:const TextStyle(fontWeight:FontWeight.bold)),children:ch));}
