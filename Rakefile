require "rubygems"
require "bundler/setup"
require "stringex"
require_relative "misc/tools/gist.rb"

## -- Rsync Deploy config -- ##
# Be sure your public key is listed in your server's ~/.ssh/authorized_keys file
ssh_user       = "user@domain.com"
ssh_port       = "22"
document_root  = "~/website.com/"
rsync_delete   = false
rsync_args     = ""  # Any extra arguments to pass to rsync
deploy_default = "push"

# This will be configured for you when you run config_deploy
deploy_branch  = "master"

## -- Misc Configs -- ##

public_dir      = "public"    # compiled site directory
source_dir      = "source"    # source file directory
blog_index_dir  = 'source'    # directory for your blog's index page (if you put your index in source/blog/index.html, set this to 'source/blog')
deploy_dir      = "_deploy"   # deploy directory (for Github pages deployment)
stash_dir       = "_stash"    # directory to stash posts for speedy generation
posts_dir       = "_posts"    # directory for blog files
printables_dir   = "assets/printables"# directory for printable version blog files
miscs_dir   = "assets/miscs"# directory for printable version blog files
themes_dir      = ".themes"   # directory for blog files
new_post_ext    = "markdown"  # default new post file extension when using the new_post task
new_page_ext    = "markdown"  # default new page file extension when using the new_page task
server_port     = "4000"      # port for preview server eg. localhost:4000

blog_url = "http://wantee.github.io"

if (/cygwin|mswin|mingw|bccwin|wince|emx/ =~ RUBY_PLATFORM) != nil
  puts '## Set the codepage to 65001 for Windows machines'
  `chcp 65001`
end

desc "Initial setup for Octopress: copies the default theme into the path of Jekyll's generator. Rake install defaults to rake install[classic] to install a different theme run rake install[some_theme_name]"
task :install, :theme do |t, args|
  if File.directory?(source_dir) || File.directory?("sass")
    abort("rake aborted!") if ask("A theme is already installed, proceeding will overwrite existing files. Are you sure?", ['y', 'n']) == 'n'
  end
  # copy theme into working Jekyll directories
  theme = args.theme || 'classic'
  puts "## Copying "+theme+" theme into ./#{source_dir} and ./sass"
  mkdir_p source_dir
  cp_r "#{themes_dir}/#{theme}/source/.", source_dir
  mkdir_p "sass"
  cp_r "#{themes_dir}/#{theme}/sass/.", "sass"
  mkdir_p "#{source_dir}/#{posts_dir}"
  mkdir_p public_dir
end

#######################
# Working with Jekyll #
#######################

posts = FileList["#{source_dir}/#{posts_dir}/*.markdown"]
pdfs = Array.new()

posts.each do |post|
  pdf = post.sub(/\/#{posts_dir}\//, "/#{printables_dir}/")
  pdf = pdf.sub(/\.markdown$/, ".pdf")
  pdfs.push(pdf)
  file pdf => post do |t|
    puts "Converting #{t.prerequisites.first} to #{t.name}"
    gen_pdf("#{t.prerequisites.first}", "#{t.name}", source_dir, posts_dir, blog_url)
  end
end

desc "Generate pdfs"
task :gen_pdf => pdfs do
  pdfs.each do |pdf|
    Rake::Task[pdf].execute
  end
end

bp_dir = "misc/BP-doc/"
bp = "#{source_dir}/#{posts_dir}/2015-03-11-note-on-learning-neural-network.markdown"
desc "Generate BP-doc"
file bp => ["#{bp_dir}/BP.toc", "#{bp_dir}/BP.hst"] do |t|
  version = 0.1
  File.open("#{bp_dir}/BP.hst", 'r') do |f|
    while line = f.gets
      if /\\def\s+\\vhCurrentVersion\s+{(?<version>.*?)}/ =~ line
        break
      end
    end
  end
  puts version
  if ! File.exists?("#{source_dir}/#{miscs_dir}/")
    mkdir_p "#{source_dir}/#{miscs_dir}/"
  end
  bp_dst = "#{miscs_dir}/BP-#{version}.pdf"
  system("cp #{bp_dir}/BP.pdf #{source_dir}/#{bp_dst}")

  contents = ""
  File.open("#{bp_dir}/BP.toc", 'r') do |f|
    while line = f.gets
      if /{\\numberline {(?<num>.*?)}(?<title>.*?)}/ =~ line
        if /\\contentsline {section}/ =~ line
          contents = "#{contents}* #{num} #{title}\n"
        end
        if /\\contentsline {subsection}/ =~ line
          contents = "#{contents}  * #{num} #{title}\n"
        end
        if /\\contentsline {subsubsection}/ =~ line
          contents = "#{contents}    * #{num} #{title}\n"
        end
      end
    end
  end

  File.open("#{bp_dir}/markdown.tmp", 'w') do |tmp|
    File.open("#{bp}", 'r') do |f|
      skip = false
      while line = f.gets
        if /{% comment %} FOR-TOC {% endcomment %}/ =~ line
          tmp.puts line
          tmp.puts contents
          skip = true
          next
        end

        if /{% comment %} FOR-TOC-END {% endcomment %}/ =~ line
          skip = false
        end

        if skip
          next
        end

        if /{% comment %} FOR-PDFLINK {% endcomment %}(.*?){% comment %} FOR-PDFLINK-END {% endcomment %}/ =~ line
          line = line.sub(/{% comment %} FOR-PDFLINK {% endcomment %}(.*?){% comment %} FOR-PDFLINK-END {% endcomment %}/, 
                  "{% comment %} FOR-PDFLINK {% endcomment %}#{blog_url}/#{bp_dst}{% comment %} FOR-PDFLINK-END {% endcomment %}")
        end
        tmp.puts line
      end
    end
  end
  system("mv #{bp_dir}/markdown.tmp #{bp}")
end

desc "Generate BP-doc toc"
file "#{bp_dir}/BP.toc" => "#{bp_dir}/BP.tex" do
  system("cd #{bp_dir}; make > /dev/null")
end

desc "Generate BP-doc hst"
file "#{bp_dir}/BP.hst" => "#{bp_dir}/BP.tex" do
  system("cd #{bp_dir}; make > /dev/null")
end

desc "Generate jekyll site"
task :generate => pdfs do
  raise "### You haven't set anything up yet. First run `rake install` to set up an Octopress theme." unless File.directory?(source_dir)
  puts "## Generating Site with Jekyll"
  system "compass compile --css-dir #{source_dir}/stylesheets"
  system "jekyll build"
end

desc "Watch the site and regenerate when it changes"
task :watch do
  raise "### You haven't set anything up yet. First run `rake install` to set up an Octopress theme." unless File.directory?(source_dir)
  puts "Starting to watch source with Jekyll and Compass."
  system "compass compile --css-dir #{source_dir}/stylesheets" unless File.exist?("#{source_dir}/stylesheets/screen.css")
  jekyllPid = Process.spawn({"OCTOPRESS_ENV"=>"preview"}, "jekyll build --watch")
  compassPid = Process.spawn("compass watch")

  trap("INT") {
    [jekyllPid, compassPid].each { |pid| Process.kill(9, pid) rescue Errno::ESRCH }
    exit 0
  }

  [jekyllPid, compassPid].each { |pid| Process.wait(pid) }
end

desc "preview the site in a web browser"
task :preview do
  raise "### You haven't set anything up yet. First run `rake install` to set up an Octopress theme." unless File.directory?(source_dir)
  puts "Starting to watch source with Jekyll and Compass. Starting Rack on port #{server_port}"
  system "compass compile --css-dir #{source_dir}/stylesheets" unless File.exist?("#{source_dir}/stylesheets/screen.css")
  jekyllPid = Process.spawn({"OCTOPRESS_ENV"=>"preview"}, "jekyll build --watch")
  compassPid = Process.spawn("compass watch")
  rackupPid = Process.spawn("rackup --port #{server_port}")

  trap("INT") {
    [jekyllPid, compassPid, rackupPid].each { |pid| Process.kill(9, pid) rescue Errno::ESRCH }
    exit 0
  }

  [jekyllPid, compassPid, rackupPid].each { |pid| Process.wait(pid) }
end

# usage rake new_post[my-new-post] or rake new_post['my new post'] or rake new_post (defaults to "new-post")
desc "Begin a new post in #{source_dir}/#{posts_dir}"
task :new_post, :title do |t, args|
  if args.title
    title = args.title
  else
    title = get_stdin("Enter a title for your post: ")
  end
  raise "### You haven't set anything up yet. First run `rake install` to set up an Octopress theme." unless File.directory?(source_dir)
  mkdir_p "#{source_dir}/#{posts_dir}"
  filename = "#{source_dir}/#{posts_dir}/#{Time.now.strftime('%Y-%m-%d')}-#{title.to_url}.#{new_post_ext}"
  if File.exist?(filename)
    abort("rake aborted!") if ask("#{filename} already exists. Do you want to overwrite?", ['y', 'n']) == 'n'
  end
  puts "Creating new post: #{filename}"
  open(filename, 'w') do |post|
    post.puts "---"
    post.puts "layout: post"
    post.puts "title: \"#{title.gsub(/&/,'&amp;')}\""
    post.puts "author: Wantee Wang"
    post.puts "date: #{Time.now.strftime('%Y-%m-%d %H:%M:%S %z')}"
    post.puts "comments: true"
    post.puts "categories: "
    post.puts "header-includes:"
    post.puts "   - \\usepackage{graphicx}"
    post.puts "   - \\usepackage[all]{hypcap}"
    post.puts "---"
  end
end

# usage rake new_page[my-new-page] or rake new_page[my-new-page.html] or rake new_page (defaults to "new-page.markdown")
desc "Create a new page in #{source_dir}/(filename)/index.#{new_page_ext}"
task :new_page, :filename do |t, args|
  raise "### You haven't set anything up yet. First run `rake install` to set up an Octopress theme." unless File.directory?(source_dir)
  args.with_defaults(:filename => 'new-page')
  page_dir = [source_dir]
  if args.filename.downcase =~ /(^.+\/)?(.+)/
    filename, dot, extension = $2.rpartition('.').reject(&:empty?)         # Get filename and extension
    title = filename
    page_dir.concat($1.downcase.sub(/^\//, '').split('/')) unless $1.nil?  # Add path to page_dir Array
    if extension.nil?
      page_dir << filename
      filename = "index"
    end
    extension ||= new_page_ext
    page_dir = page_dir.map! { |d| d = d.to_url }.join('/')                # Sanitize path
    filename = filename.downcase.to_url

    mkdir_p page_dir
    file = "#{page_dir}/#{filename}.#{extension}"
    if File.exist?(file)
      abort("rake aborted!") if ask("#{file} already exists. Do you want to overwrite?", ['y', 'n']) == 'n'
    end
    puts "Creating new page: #{file}"
    open(file, 'w') do |page|
      page.puts "---"
      page.puts "layout: page"
      page.puts "title: \"#{title}\""
      page.puts "date: #{Time.now.strftime('%Y-%m-%d %H:%M')}"
      page.puts "comments: true"
      page.puts "sharing: true"
      page.puts "footer: true"
      page.puts "---"
    end
  else
    puts "Syntax error: #{args.filename} contains unsupported characters"
  end
end

# usage rake isolate[my-post]
desc "Move all other posts than the one currently being worked on to a temporary stash location (stash) so regenerating the site happens much more quickly."
task :isolate, :filename do |t, args|
  stash_dir = "#{source_dir}/#{stash_dir}"
  FileUtils.mkdir(stash_dir) unless File.exist?(stash_dir)
  Dir.glob("#{source_dir}/#{posts_dir}/*.*") do |post|
    FileUtils.mv post, stash_dir unless post.include?(args.filename)
  end
end

desc "Move all stashed posts back into the posts directory, ready for site generation."
task :integrate do
  FileUtils.mv Dir.glob("#{source_dir}/#{stash_dir}/*.*"), "#{source_dir}/#{posts_dir}/"
end

desc "Clean out caches: .pygments-cache, .gist-cache, .sass-cache"
task :clean do
  rm_rf [Dir.glob(".pygments-cache/**"), Dir.glob(".gist-cache/**"), Dir.glob(".sass-cache/**"), "source/stylesheets/screen.css"]
end

desc "Move sass to sass.old, install sass theme updates, replace sass/custom with sass.old/custom"
task :update_style, :theme do |t, args|
  theme = args.theme || 'classic'
  if File.directory?("sass.old")
    puts "removed existing sass.old directory"
    rm_r "sass.old", :secure=>true
  end
  mv "sass", "sass.old"
  puts "## Moved styles into sass.old/"
  cp_r "#{themes_dir}/"+theme+"/sass/", "sass", :remove_destination=>true
  cp_r "sass.old/custom/.", "sass/custom/", :remove_destination=>true
  puts "## Updated Sass ##"
end

desc "Move source to source.old, install source theme updates, replace source/_includes/navigation.html with source.old's navigation"
task :update_source, :theme do |t, args|
  theme = args.theme || 'classic'
  if File.directory?("#{source_dir}.old")
    puts "## Removed existing #{source_dir}.old directory"
    rm_r "#{source_dir}.old", :secure=>true
  end
  mkdir "#{source_dir}.old"
  cp_r "#{source_dir}/.", "#{source_dir}.old"
  puts "## Copied #{source_dir} into #{source_dir}.old/"
  cp_r "#{themes_dir}/"+theme+"/source/.", source_dir, :remove_destination=>true
  cp_r "#{source_dir}.old/_includes/custom/.", "#{source_dir}/_includes/custom/", :remove_destination=>true
  cp "#{source_dir}.old/favicon.png", source_dir
  mv "#{source_dir}/index.html", "#{blog_index_dir}", :force=>true if blog_index_dir != source_dir
  cp "#{source_dir}.old/index.html", source_dir if blog_index_dir != source_dir && File.exists?("#{source_dir}.old/index.html")
  puts "## Updated #{source_dir} ##"
end

##############
# Deploying  #
##############

desc "Default deploy task"
task :deploy do
  # Check if preview posts exist, which should not be published
  if File.exists?(".preview-mode")
    puts "## Found posts in preview mode, regenerating files ..."
    File.delete(".preview-mode")
    Rake::Task[:generate].execute
  end

  Rake::Task[:copydot].invoke(source_dir, public_dir)
  Rake::Task["#{deploy_default}"].execute
end

desc "Generate website and deploy"
task :gen_deploy => [:integrate, :generate, :deploy] do
end

desc "copy dot files for deployment"
task :copydot, :source, :dest do |t, args|
  FileList["#{args.source}/**/.*"].exclude("**/.", "**/..", "**/.DS_Store", "**/._*").each do |file|
    cp_r file, file.gsub(/#{args.source}/, "#{args.dest}") unless File.directory?(file)
  end
end

desc "Deploy website via rsync"
task :rsync do
  exclude = ""
  if File.exists?('./rsync-exclude')
    exclude = "--exclude-from '#{File.expand_path('./rsync-exclude')}'"
  end
  puts "## Deploying website via Rsync"
  ok_failed system("rsync -avze 'ssh -p #{ssh_port}' #{exclude} #{rsync_args} #{"--delete" unless rsync_delete == false} #{public_dir}/ #{ssh_user}:#{document_root}")
end

desc "deploy public directory to github pages"
multitask :push do
  puts "## Deploying branch to Github Pages "
  puts "## Pulling any updates from Github Pages "
  cd "#{deploy_dir}" do 
    Bundler.with_clean_env { system "git pull" }
  end
  (Dir["#{deploy_dir}/*"]).each { |f| rm_rf(f) }
  Rake::Task[:copydot].invoke(public_dir, deploy_dir)
  puts "\n## Copying #{public_dir} to #{deploy_dir}"
  cp_r "#{public_dir}/.", deploy_dir
  cd "#{deploy_dir}" do
    system "git add -A"
    message = "Site updated at #{Time.now.utc}"
    puts "\n## Committing: #{message}"
    system "git commit -m \"#{message}\""
    puts "\n## Pushing generated #{deploy_dir} website"
    Bundler.with_clean_env { system "git push origin #{deploy_branch}" }
    puts "\n## Github Pages deploy complete"
  end
end

desc "Update configurations to support publishing to root or sub directory"
task :set_root_dir, :dir do |t, args|
  puts ">>> !! Please provide a directory, eg. rake config_dir[publishing/subdirectory]" unless args.dir
  if args.dir
    if args.dir == "/"
      dir = ""
    else
      dir = "/" + args.dir.sub(/(\/*)(.+)/, "\\2").sub(/\/$/, '');
    end
    rakefile = IO.read(__FILE__)
    rakefile.sub!(/public_dir(\s*)=(\s*)(["'])[\w\-\/]*["']/, "public_dir\\1=\\2\\3public#{dir}\\3")
    File.open(__FILE__, 'w') do |f|
      f.write rakefile
    end
    compass_config = IO.read('config.rb')
    compass_config.sub!(/http_path(\s*)=(\s*)(["'])[\w\-\/]*["']/, "http_path\\1=\\2\\3#{dir}/\\3")
    compass_config.sub!(/http_images_path(\s*)=(\s*)(["'])[\w\-\/]*["']/, "http_images_path\\1=\\2\\3#{dir}/images\\3")
    compass_config.sub!(/http_fonts_path(\s*)=(\s*)(["'])[\w\-\/]*["']/, "http_fonts_path\\1=\\2\\3#{dir}/fonts\\3")
    compass_config.sub!(/css_dir(\s*)=(\s*)(["'])[\w\-\/]*["']/, "css_dir\\1=\\2\\3public#{dir}/stylesheets\\3")
    File.open('config.rb', 'w') do |f|
      f.write compass_config
    end
    jekyll_config = IO.read('_config.yml')
    jekyll_config.sub!(/^destination:.+$/, "destination: public#{dir}")
    jekyll_config.sub!(/^subscribe_rss:\s*\/.+$/, "subscribe_rss: #{dir}/atom.xml")
    jekyll_config.sub!(/^root:.*$/, "root: /#{dir.sub(/^\//, '')}")
    File.open('_config.yml', 'w') do |f|
      f.write jekyll_config
    end
    rm_rf public_dir
    mkdir_p "#{public_dir}#{dir}"
    puts "## Site's root directory is now '/#{dir.sub(/^\//, '')}' ##"
  end
end

desc "Set up _deploy folder and deploy branch for Github Pages deployment"
task :setup_github_pages, :repo do |t, args|
  if args.repo
    repo_url = args.repo
  else
    puts "Enter the read/write url for your repository"
    puts "(For example, 'git@github.com:your_username/your_username.github.io.git)"
    puts "           or 'https://github.com/your_username/your_username.github.io')"
    repo_url = get_stdin("Repository url: ")
  end
  protocol = (repo_url.match(/(^git)@/).nil?) ? 'https' : 'git'
  if protocol == 'git'
    user = repo_url.match(/:([^\/]+)/)[1]
  else
    user = repo_url.match(/github\.com\/([^\/]+)/)[1]
  end
  branch = (repo_url.match(/\/[\w-]+\.github\.(?:io|com)/).nil?) ? 'gh-pages' : 'master'
  project = (branch == 'gh-pages') ? repo_url.match(/([^\/]+?)(\.git|$)/i)[1] : ''
  unless (`git remote -v` =~ /origin.+?octopress(?:\.git)?/).nil?
    # If octopress is still the origin remote (from cloning) rename it to octopress
    system "git remote rename origin octopress"
    if branch == 'master'
      # If this is a user/organization pages repository, add the correct origin remote
      # and checkout the source branch for committing changes to the blog source.
      system "git remote add origin #{repo_url}"
      puts "Added remote #{repo_url} as origin"
      system "git config branch.master.remote origin"
      puts "Set origin as default remote"
      system "git branch -m master source"
      puts "Master branch renamed to 'source' for committing your blog source files"
    else
      unless !public_dir.match("#{project}").nil?
        system "rake set_root_dir[#{project}]"
      end
    end
  end
  url = post_url(user, project, source_dir)
  jekyll_config = IO.read('_config.yml')
  jekyll_config.sub!(/^url:.*$/, "url: #{url}")
  File.open('_config.yml', 'w') do |f|
    f.write jekyll_config
  end
  rm_rf deploy_dir
  mkdir deploy_dir
  cd "#{deploy_dir}" do
    system "git init"
    system 'echo "My Octopress Page is coming soon &hellip;" > index.html'
    system "git add ."
    system "git commit -m \"Octopress init\""
    system "git branch -m gh-pages" unless branch == 'master'
    system "git remote add origin #{repo_url}"
    rakefile = IO.read(__FILE__)
    rakefile.sub!(/deploy_branch(\s*)=(\s*)(["'])[\w-]*["']/, "deploy_branch\\1=\\2\\3#{branch}\\3")
    rakefile.sub!(/deploy_default(\s*)=(\s*)(["'])[\w-]*["']/, "deploy_default\\1=\\2\\3push\\3")
    File.open(__FILE__, 'w') do |f|
      f.write rakefile
    end
  end
  puts "\n---\n## Now you can deploy to #{repo_url} with `rake deploy` ##"
end

def ok_failed(condition)
  if (condition)
    puts "OK"
  else
    puts "FAILED"
  end
end

def get_stdin(message)
  print message
  STDIN.gets.chomp
end

def ask(message, valid_options)
  if valid_options
    answer = get_stdin("#{message} #{valid_options.to_s.gsub(/"/, '').gsub(/, /,'/')} ") while !valid_options.include?(answer)
  else
    answer = get_stdin(message)
  end
  answer
end

def post_url(user, project, source_dir)
  cname = "#{source_dir}/CNAME"
  url = if File.exists?(cname)
    "http://#{IO.read(cname).strip}"
  else
    "http://#{user.downcase}.github.io"
  end
  url += "/#{project}" unless project == ''
  url
end

def gen_pdf(markdownfile, pdffile, source_dir, posts_dir, blog_url)
  pdfdir = File.dirname(pdffile)
  if ! File.exists?(pdfdir)
    mkdir_p pdfdir
  end

  obib="#{source_dir}/_bibliography/references"
  bib="#{pdfdir}/references"
  has_bib = false
  has_gist = false

  tmpfile="#{pdffile}.markdown"
  comment = false
  File.open(tmpfile, 'w') do |post|
    File.open(markdownfile, "r") do |file|  
      while line = file.gets
        line = line.strip

        if /^{% comment %}$/ =~ line
          comment = true
          next
        end

        if /^{% endcomment %}$/ =~ line
          comment = false
          next
        end

        if comment
          next
        end

        line=line.gsub(/\$\$\s*\\begin{equation}/, '\\begin{equation}')
        line=line.gsub(/\\end{equation}\s*\$\$/, '\\end{equation}')
        line=line.gsub(/\\\*/, '*')
        line=line.gsub(/\\\|/, '|')
        line=line.gsub(/\\_/, '_')

        line=line.gsub(/\* list element with functor item/, '')
        line=line.gsub(/{:toc}/, '\tableofcontents')

        line=line.sub(/^#(#*)/, '\1')

        if /{% img (?<markup>.*) %}/ =~ line
          @img = get_img_label(markup)
          line="\\begin{figure}[h]\\centering\\includegraphics[width=\\textwidth]{#{source_dir}/#{@img['src']}}\\caption{#{@img['title']}}\\label{#{@img['alt']}}\\end{figure}"
        end

        while /{% comment %} FOR-LATEX (?<markup>.*) {% endcomment %}/ =~ line
          line = line.sub(/{% comment %} FOR-LATEX (.*?) {% endcomment %}/, markup)
        end

        line = line.gsub(/{% comment %} (.*?) {% endcomment %}/, "")

        while /{% post_link (?<markup>[^\s]+)(?<text>\s+.+)? %}/ =~ line
          /^(?<year>\d+)-(?<month>\d+)-(?<day>\d+)-(?<title>.*)/ =~ markup

          if ! text
            File.open("#{source_dir}/#{posts_dir}/#{markup}.markdown", 'r') do |f|
                while l = f.gets
                  if /title: (?:"|')(?<text>.*)(?:"|')/ =~ l
                    break
                  end
                end
            end
          end
          line = line.sub(/{% post_link (.*?) %}/, "\\href{#{blog_url}/blog/#{year}/#{month}/#{day}/#{title}/}{#{text}}")
        end

        if /##*\s*References/ =~ line
          next
        end

        if /{% bibliography .*? %}/ =~ line
          line="\\bibliographystyle{unsrt}\\bibliography{#{bib}}"
          has_bib=true

          gen_bib("#{obib}.bib", "#{bib}.bib")
        end

        while /{% cite\s+(?<citation>.*?)\s+%}/ =~ line
          citation=citation.sub(/\s+/, ',')
          line=line.sub(/{% cite\s+(.*?)\s+%}/, "\\cite{#{citation}}")
        end

        if /{%\s+gist\s+(?<gist_txt>.*?)\s+%}/ =~ line
          has_gist=true
          gist = Gist.new(gist_txt)
          gist_file = gist.render()
          if gist_file == ""
            line = ""
          else
            lang = "text"
            if /\.py$/ =~ gist_file
              lang = 'Python'
            end

            line = "\\inputminted[mathescape, linenos, frame=lines, framesep=2mm]{#{lang}}{#{gist_file}}"
          end
        end

        post.puts line
      end  
    end
  end 

  texfile=pdffile.sub(/.pdf$/, '.tex')
  base=pdffile.sub(/.pdf$/, '')
  pkgfile="#{pdfdir}/header.tex"
  system "echo > #{pkgfile}"
  if has_gist
    system "echo \"\\usepackage{minted}\" >> #{pkgfile}"
  end

  if has_bib
    system "echo \"\\usepackage[sort&compress, numbers]{natbib}\" >> #{pkgfile}"

    system "pandoc -s --include-in-header=#{pkgfile} #{tmpfile} -o #{texfile} "
	system "xelatex -output-directory=#{pdfdir} -no-pdf --interaction=nonstopmode #{base} >/dev/null"
	system "bibtex #{base} >/dev/null"
	system "xelatex -output-directory=#{pdfdir} -no-pdf --interaction=nonstopmode #{base} >/dev/null"
	system "xelatex -output-directory=#{pdfdir} --interaction=nonstopmode #{base} >/dev/null"

    system "rm -rf #{bib}"
	system "rm -rf #{pdfdir}/*.aux"
	system "rm -rf #{pdfdir}/*.log"
	system "rm -rf #{pdfdir}/*.lot"
	system "rm -rf #{pdfdir}/*.out"
	system "rm -rf #{pdfdir}/*.toc"
	system "rm -rf #{pdfdir}/*.blg"
	system "rm -rf #{pdfdir}/*.bbl"
	system "rm -rf #{pdfdir}/*.lof"
	system "rm -rf #{pdfdir}/*.xdv"
	system "rm -rf #{pdfdir}/*.hst"
	system "rm -rf #{pdfdir}/*.ver"
	system "rm -rf #{pdfdir}/*.synctex.gz"
  else
    system "pandoc --include-in-header=#{pkgfile} --latex-engine=xelatex -N #{tmpfile} -o #{pdffile}"
  end

  system "rm -rf #{pkgfile}"
  system "rm -rf #{tmpfile}"
  system "rm -rf input.pyg"
end

# from image_tag plugin
def get_img_label(markup)
  @img = nil
  attributes = ['class', 'src', 'width', 'height', 'title']

  if markup =~ /(?<class>\S.*\s+)?(?<src>(?:https?:\/\/|\/|\S+\/)\S+)(?:\s+(?<width>\d+))?(?:\s+(?<height>\d+))?(?<title>\s+.+)?/i
    @img = attributes.reduce({}) { |img, attr| img[attr] = $~[attr].strip if $~[attr]; img }
    if /(?:"|')(?<title>[^"']+)?(?:"|')\s+(?:"|')(?<alt>[^"']+)?(?:"|')/ =~ @img['title']
      @img['title']  = title
      @img['alt']    = alt
    else
#@img['alt']    = @img['title'].gsub!(/"/, '&#34;') if @img['title']
      @img['alt']    = @img['title']
    end
    @img['class'].gsub!(/"/, '') if @img['class']
  end
  @img
end

def gen_bib(obib, bib)
  File.open(obib, 'r') do |f|
    File.open(bib, 'w') do |o|
      while l = f.gets
        if /<a\s+href="?(?<url>.*?)"?\s*>(?<text>.*?)<\/a>/ =~ l
          l = l.sub(/<a\s+href=(.*?)>(.*?)<\/a>/, "\\href{#{url}}{#{text}}")
        end

        o.puts l
      end
    end
  end
end

desc "list tasks"
task :list do
  puts "Tasks: #{(Rake::Task.tasks - [Rake::Task[:list]]).join(', ')}"
  puts "(type rake -T for more detail)\n\n"
end
